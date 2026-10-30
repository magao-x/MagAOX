/** \file dmSaturation_test.cpp
 * \brief Deterministic burst, validation and concurrent tests of saturation payload accumulation.
 * \ingroup dmSaturation_unit_test
 */
#include "../../../../tests/testXWC.hpp"
#include "../dmSaturation.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace libXWCTest
{
namespace dmSaturationTests
{

using Accumulator = MagAOX::app::dev::dmSaturation;

/// \cond DM_SATURATION_TEST_HELPERS
namespace
{

/// Expose impossible-to-reach production-scale totals only for deterministic overflow validation.
class OverflowAccumulator : public Accumulator
{
  public:
    /// Set a pending sample count and pixel total while neither owner is running.
    void setPending( uint64_t samples /**< [in] Number of pending masks. */,
                     uint64_t first /**< [in] Pending total for the first actuator. */ );
};

void OverflowAccumulator::setPending( uint64_t samples, uint64_t first )
{
    m_samples    = samples;
    m_pending[0] = first;
}

} // namespace
/// \endcond

/// Every mask contributes exactly once, independently of delayed or duplicate notifications.
TEST_CASE( "DM saturation preserves distinct burst payloads", "[dev::dmSaturation]" )
{
    // clang-format off
    #ifdef DMSATURATION_TEST_DOXYGEN_REF
       MagAOX::app::dev::dmSaturation::configure;
       MagAOX::app::dev::dmSaturation::add;
       MagAOX::app::dev::dmSaturation::drain;
       MagAOX::app::dev::dmSaturation::pending;
    #endif
    // clang-format on

    Accumulator accumulator;
    accumulator.configure( 3 );
    const std::array<uint8_t, 3> first{ 1, 0, 1 };
    const std::array<uint8_t, 3> second{ 0, 1, 1 };
    const std::array<uint8_t, 3> third{ 1, 0, 0 };
    std::array<uint64_t, 3>      totals{ 5, 6, 7 };

    REQUIRE_FALSE( accumulator.pending() );
    REQUIRE( accumulator.add( first.data(), first.size() ) );
    REQUIRE( accumulator.add( second.data(), second.size() ) );
    REQUIRE( accumulator.add( third.data(), third.size() ) );
    REQUIRE( accumulator.pending() );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 3 );
    REQUIRE( totals == std::array<uint64_t, 3>{ 7, 7, 9 } );
    REQUIRE_FALSE( accumulator.pending() );

    for( size_t notification = 0; notification < 4; ++notification )
    {
        REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 0 );
        REQUIRE( totals == std::array<uint64_t, 3>{ 7, 7, 9 } );
    }

    REQUIRE( accumulator.add( second.data(), second.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 1 );
    REQUIRE( totals == std::array<uint64_t, 3>{ 7, 8, 10 } );
}

/// Invalid masks and drain destinations preserve previously accumulated samples and caller totals.
TEST_CASE( "DM saturation rejects malformed operations atomically", "[dev::dmSaturation]" )
{
    // clang-format off
    #ifdef DMSATURATION_TEST_DOXYGEN_REF
       MagAOX::app::dev::dmSaturation::configure;
       MagAOX::app::dev::dmSaturation::add;
       MagAOX::app::dev::dmSaturation::drain;
    #endif
    // clang-format on

    Accumulator                  accumulator;
    const std::array<uint8_t, 3> valid{ 1, 0, 1 };
    const std::array<uint8_t, 3> invalid{ 1, 1, 2 };
    std::array<uint64_t, 3>      totals{ 2, 3, 4 };

    REQUIRE_FALSE( accumulator.add( valid.data(), valid.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 0 );
    accumulator.configure( valid.size() );
    REQUIRE( accumulator.add( valid.data(), valid.size() ) );
    REQUIRE_FALSE( accumulator.add( nullptr, valid.size() ) );
    REQUIRE_FALSE( accumulator.add( valid.data(), 0 ) );
    REQUIRE_FALSE( accumulator.add( valid.data(), valid.size() - 1 ) );
    REQUIRE_FALSE( accumulator.add( invalid.data(), invalid.size() ) );
    REQUIRE( accumulator.drain( nullptr, totals.size() ) == 0 );
    REQUIRE( accumulator.drain( totals.data(), 0 ) == 0 );
    REQUIRE( accumulator.drain( totals.data(), totals.size() - 1 ) == 0 );
    REQUIRE( accumulator.pending() );
    REQUIRE( totals == std::array<uint64_t, 3>{ 2, 3, 4 } );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 1 );
    REQUIRE( totals == std::array<uint64_t, 3>{ 3, 3, 5 } );

    accumulator.configure( 0 );
    REQUIRE_FALSE( accumulator.pending() );
    REQUIRE_FALSE( accumulator.add( nullptr, 0 ) );
    REQUIRE_FALSE( accumulator.add( valid.data(), valid.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 0 );
}

/// Overflow never partially modifies a mask, pending payload or caller-owned reporting window.
TEST_CASE( "DM saturation retains payload on overflow", "[dev::dmSaturation]" )
{
    // clang-format off
    #ifdef DMSATURATION_TEST_DOXYGEN_REF
       MagAOX::app::dev::dmSaturation::add;
       MagAOX::app::dev::dmSaturation::drain;
    #endif
    // clang-format on

    OverflowAccumulator accumulator;
    accumulator.configure( 2 );
    constexpr auto               limit = std::numeric_limits<uint64_t>::max();
    const std::array<uint8_t, 2> both{ 1, 1 };
    std::array<uint64_t, 2>      totals{ 0, 0 };

    accumulator.setPending( limit, 0 );
    REQUIRE_FALSE( accumulator.add( both.data(), both.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == limit );
    REQUIRE( totals == std::array<uint64_t, 2>{ 0, 0 } );

    accumulator.setPending( 1, limit );
    REQUIRE_FALSE( accumulator.add( both.data(), both.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 1 );
    REQUIRE( totals == std::array<uint64_t, 2>{ limit, 0 } );

    totals = { 4, limit };
    REQUIRE( accumulator.add( both.data(), both.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 0 );
    REQUIRE( accumulator.pending() );
    REQUIRE( totals == std::array<uint64_t, 2>{ 4, limit } );
    totals[1] = 0;
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 1 );
    REQUIRE( totals == std::array<uint64_t, 2>{ 5, 1 } );
}

/// Clearing removes only pending payloads and preserves correct accounting for later commands.
TEST_CASE( "DM saturation clears pending masks without replaying notifications", "[dev::dmSaturation]" )
{
    // clang-format off
    #ifdef DMSATURATION_TEST_DOXYGEN_REF
       MagAOX::app::dev::dmSaturation::clear;
       MagAOX::app::dev::dmSaturation::add;
       MagAOX::app::dev::dmSaturation::drain;
    #endif
    // clang-format on

    Accumulator accumulator;
    accumulator.clear();
    accumulator.configure( 3 );
    const std::array<uint8_t, 3> oldMask{ 1, 0, 1 };
    const std::array<uint8_t, 3> newMask{ 0, 1, 0 };
    std::array<uint64_t, 3>      totals{ 4, 5, 6 };

    REQUIRE( accumulator.add( oldMask.data(), oldMask.size() ) );
    REQUIRE( accumulator.add( oldMask.data(), oldMask.size() ) );
    accumulator.clear();
    REQUIRE_FALSE( accumulator.pending() );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 0 );
    REQUIRE( totals == std::array<uint64_t, 3>{ 4, 5, 6 } );
    REQUIRE( accumulator.add( newMask.data(), newMask.size() ) );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 1 );
    REQUIRE( totals == std::array<uint64_t, 3>{ 4, 6, 6 } );
    REQUIRE( accumulator.drain( totals.data(), totals.size() ) == 0 );
}

/// Concurrent publication and consumption preserve all command counts and per-actuator contributions.
TEST_CASE( "DM saturation transfers concurrent bursts without payload loss", "[dev::dmSaturation]" )
{
    // clang-format off
    #ifdef DMSATURATION_TEST_DOXYGEN_REF
       MagAOX::app::dev::dmSaturation::add;
       MagAOX::app::dev::dmSaturation::drain;
    #endif
    // clang-format on

    Accumulator accumulator;
    accumulator.configure( 4 );
    constexpr uint64_t      samples    = 10000;
    constexpr uint64_t      firstBatch = 128;
    std::atomic<bool>       done{ false };
    std::atomic<bool>       accepted{ true };
    std::array<uint64_t, 4> totals{};
    std::mutex              phaseMutex;
    std::condition_variable changed;
    bool                    firstPublished     = false;
    bool                    continuePublishing = false;
    uint64_t                drained            = 0;

    std::thread producer(
        [&]()
        {
            for( uint64_t index = 0; index < samples; ++index )
            {
                const std::array<uint8_t, 4> mask{ 1, uint8_t( index % 2 ), uint8_t( index % 3 == 0 ), 0 };

                if( !accumulator.add( mask.data(), mask.size() ) )
                {
                    accepted.store( false, std::memory_order_relaxed );
                }

                if( index + 1 == firstBatch )
                {
                    std::unique_lock<std::mutex> lock( phaseMutex );
                    firstPublished = true;
                    changed.notify_one();
                    changed.wait( lock,
                                  [&]()
                                  {
                                      return continuePublishing;
                                  } );
                }
            }

            done.store( true, std::memory_order_release );
        } );
    bool firstReady = false;

    { //mutex scope
        std::unique_lock<std::mutex> lock( phaseMutex );
        firstReady = changed.wait_for( lock,
                                       std::chrono::seconds( 3 ),
                                       [&]()
                                       {
                                           return firstPublished;
                                       } );
    }

    const uint64_t firstDrained = accumulator.drain( totals.data(), totals.size() );
    drained += firstDrained;

    { //mutex scope
        std::lock_guard<std::mutex> lock( phaseMutex );
        continuePublishing = true;
    }

    changed.notify_one();

    while( !done.load( std::memory_order_acquire ) )
    {
        drained += accumulator.drain( totals.data(), totals.size() );
        std::this_thread::yield();
    }

    producer.join();
    drained += accumulator.drain( totals.data(), totals.size() );
    REQUIRE( firstReady );
    REQUIRE( firstDrained == firstBatch );
    REQUIRE( accepted.load( std::memory_order_relaxed ) );
    REQUIRE( drained == samples );
    REQUIRE( totals == std::array<uint64_t, 4>{ samples, samples / 2, ( samples + 2 ) / 3, 0 } );
    REQUIRE_FALSE( accumulator.pending() );
}

} // namespace dmSaturationTests
} // namespace libXWCTest
