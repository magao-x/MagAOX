/** \file spscQueue_test.cpp
 * \brief Ownership, capacity-loss, wrap and concurrent payload tests for the common handoff.
 */
#include "../../../tests/testXWC.hpp"
#include "../spscQueue.hpp"
#include "../timingStats.hpp"
#include "../dmTiming.hpp"

#include <atomic>
#include <thread>
#include <type_traits>

namespace libXWCTest
{
namespace spscQueueTest
{
/// \cond SPSC_TEST_HELPERS
/// Multiword payload detects torn copies as well as lost ordering.
struct Payload
{
    uint64_t sequence{ 0 };   ///< Source publication number.
    uint64_t complement{ 0 }; ///< Independently checked payload word.
    uint64_t multiple{ 0 };   ///< Additional payload word on the same publication.
};
/// \endcond

/// Verify full queues never overwrite unread slots and storage wraps without moving ownership.
/** \ingroup spscQueue_unit_test
 * \see MagAOX::spscQueue
 */
TEST_CASE( "SPSC queue preserves unread ownership and counts losses", "[spscQueue]" )
{
    MagAOX::spscQueue<Payload> queue;
    queue.configure( 3 );
    REQUIRE( queue.capacity() == 3 );
    REQUIRE( queue.queued() == 0 );

    for( uint64_t index = 1; index <= 3; ++index )
    {
        REQUIRE( queue.push( { index, ~index, index * 17 } ) );
    }

    REQUIRE_FALSE( queue.push( { 99, 0, 0 } ) );
    REQUIRE( queue.dropped() == 1 );
    Payload value;
    REQUIRE( queue.pop( value ) );
    REQUIRE( value.sequence == 1 );
    REQUIRE( queue.push( { 4, ~uint64_t( 4 ), 68 } ) );

    for( uint64_t index = 2; index <= 4; ++index )
    {
        REQUIRE( queue.pop( value ) );
        REQUIRE( value.sequence == index );
        REQUIRE( value.complement == ~index );
        REQUIRE( value.multiple == index * 17 );
    }

    REQUIRE_FALSE( queue.pop( value ) );
    queue.configure( 0 );
    REQUIRE_FALSE( queue.push( {} ) );
    REQUIRE_FALSE( queue.pop( value ) );
    REQUIRE( queue.dropped() == 1 );
}

/// Exercise independent producer and consumer ownership with multiword payload verification.
/** \ingroup spscQueue_unit_test
 * \see MagAOX::spscQueue
 */
TEST_CASE( "SPSC concurrent publication never exposes a torn or overwritten payload", "[spscQueue]" )
{
    MagAOX::spscQueue<Payload> queue;
    queue.configure( 31 );
    std::atomic<bool>     done{ false };
    std::atomic<uint64_t> accepted{ 0 };
    constexpr uint64_t    attempts = 100000;
    std::thread           producer(
        [&]()
        {
            for( uint64_t index = 1; index <= attempts; ++index )
            {
                if( queue.push( { index, ~index, index * 17 } ) )
                {
                    accepted.fetch_add( 1, std::memory_order_relaxed );
                }
            }

            done.store( true, std::memory_order_release );
        } );
    uint64_t previous = 0;
    uint64_t received = 0;
    bool     coherent = true;
    Payload  value;

    while( !done.load( std::memory_order_acquire ) || queue.queued() )
    {
        if( queue.pop( value ) )
        {
            coherent = coherent && value.sequence > previous && value.complement == ~value.sequence &&
                       value.multiple == value.sequence * 17;
            previous = value.sequence;
            ++received;
        }
        else
        {
            std::this_thread::yield();
        }
    }

    producer.join();
    REQUIRE( coherent );
    REQUIRE( received == accepted.load() );
    REQUIRE( received + queue.dropped() == attempts );
}

/// Preserve the DM population ABI while using neutral numerical accumulation.
/** \ingroup spscQueue_unit_test
 * \see MagAOX::timing::Accumulator
 * \see MagAOX::dmTiming::Stats
 */
TEST_CASE( "Neutral timing populations preserve DM statistics and signed changes", "[spscQueue]" )
{
    STATIC_REQUIRE( std::is_same_v<MagAOX::timing::Stats, MagAOX::dmTiming::Stats> );
    MagAOX::timing::Accumulator accumulator;
    REQUIRE_FALSE( accumulator.stats().valid );
    accumulator.add( -2e-6, 100, 200 );
    REQUIRE( accumulator.stats().count == 1 );
    REQUIRE_FALSE( accumulator.stats().valid );
    accumulator.add( 4e-6, 300, 400 );
    const auto result = accumulator.stats();
    REQUIRE( result.count == 2 );
    REQUIRE( result.mean == Approx( 1e-6 ) );
    REQUIRE( result.stddev == Approx( std::sqrt( 18e-12 ) ) );
    REQUIRE( result.min == -2e-6 );
    REQUIRE( result.max == 4e-6 );
    REQUIRE( result.span == Approx( 300e-9 ) );
    REQUIRE( result.valid );
}
} // namespace spscQueueTest
} // namespace libXWCTest
