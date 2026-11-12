/** \file latestSnapshot_test.cpp
 * \brief Latest-state coalescing and concurrent payload ownership tests.
 */
#include "../../../tests/testXWC.hpp"
#include "../latestSnapshot.hpp"

#include <atomic>
#include <chrono>
#include <thread>
#include <type_traits>

namespace libXWCTest
{
namespace latestSnapshotTest
{

/// \cond LATEST_SNAPSHOT_TEST_HELPERS
/// Multiword state makes torn copies visible while the producer overtakes a slow consumer.
struct Payload
{
    uint64_t                  m_sequence{ 0 };   ///< Monotonic source identity, including valid zero.
    std::array<uint64_t, 512> m_words{};         ///< Independently checked words spread over many cache lines.
    uint64_t                  m_complement{ 0 }; ///< Trailing identity word checked after the copied array.
};

/// Create a deterministic full-size state for one publication.
Payload makePayload( uint64_t sequence /**< [in] Publication identity. */ );

/// Check every copied word against its frozen publication identity.
bool coherent( const Payload &value /**< [in] Consumer-owned snapshot. */ );

Payload makePayload( uint64_t sequence )
{
    Payload value;
    value.m_sequence   = sequence;
    value.m_complement = ~sequence;

    for( size_t index = 0; index < value.m_words.size(); ++index )
    {
        value.m_words[index] = sequence ^ ( uint64_t( index + 1 ) * 0x9e3779b97f4a7c15ULL );
    }

    return value;
}

bool coherent( const Payload &value )
{
    if( value.m_complement != ~value.m_sequence )
    {
        return false;
    }

    for( size_t index = 0; index < value.m_words.size(); ++index )
    {
        if( value.m_words[index] != ( value.m_sequence ^ ( uint64_t( index + 1 ) * 0x9e3779b97f4a7c15ULL ) ) )
        {
            return false;
        }
    }

    return true;
}
/// \endcond

/// Keep empty reads nonmutating and distinguish a published zero identity from no publication.
/** \ingroup latestSnapshot_unit_test
 * \see MagAOX::latestSnapshot
 */
TEST_CASE( "Latest snapshot starts empty and preserves consumer output", "[latestSnapshot]" )
{
    MagAOX::latestSnapshot<Payload> snapshot;
    Payload                         value = makePayload( 42 );
    STATIC_REQUIRE( noexcept( snapshot.publish( value ) ) );
    STATIC_REQUIRE( noexcept( snapshot.consume( value ) ) );
    STATIC_REQUIRE_FALSE( std::is_copy_constructible_v<decltype( snapshot )> );

    CHECK_FALSE( snapshot.consume( value ) );
    CHECK( value.m_sequence == 42 );
    CHECK( coherent( value ) );
    snapshot.publish( makePayload( 0 ) );
    REQUIRE( snapshot.consume( value ) );
    CHECK( value.m_sequence == 0 );
    CHECK( coherent( value ) );
    CHECK_FALSE( snapshot.consume( value ) );
    CHECK( value.m_sequence == 0 );
    CHECK( coherent( value ) );
}

/// Coalesce unread states without a queue backlog and repeatedly exchange all three ownership roles.
/** \ingroup latestSnapshot_unit_test
 * \see MagAOX::latestSnapshot
 */
TEST_CASE( "Latest snapshot coalesces producer bursts into the newest state", "[latestSnapshot]" )
{
    MagAOX::latestSnapshot<Payload> snapshot;
    Payload                         value;

    for( uint64_t burst = 1; burst <= 20; ++burst )
    {
        for( uint64_t index = 1; index <= burst; ++index )
        {
            snapshot.publish( makePayload( burst * 100 + index ) );
        }

        REQUIRE( snapshot.consume( value ) );
        CHECK( value.m_sequence == burst * 101 );
        CHECK( coherent( value ) );
        CHECK_FALSE( snapshot.consume( value ) );
        CHECK( value.m_sequence == burst * 101 );
        CHECK( coherent( value ) );
    }
}

/// Verify independent buffer ownership with a slow consumer and final latest-state delivery.
/** \ingroup latestSnapshot_unit_test
 * \see MagAOX::latestSnapshot
 */
TEST_CASE( "Latest snapshot remains coherent and monotonic with a slow concurrent reader",
           "[latestSnapshot][concurrency]" )
{
    MagAOX::latestSnapshot<Payload> snapshot;
    constexpr uint64_t              publications = 100000;
    std::atomic<bool>               start{ false };
    std::atomic<bool>               done{ false };
    std::thread                     producer(
        [&]()
        {
            while( !start.load( std::memory_order_acquire ) )
            {
                std::this_thread::yield();
            }

            for( uint64_t sequence = 1; sequence <= publications; ++sequence )
            {
                snapshot.publish( makePayload( sequence ) );
            }

            done.store( true, std::memory_order_release );
        } );
    Payload  value;
    uint64_t previous = 0;
    uint64_t received = 0;
    bool     valid    = true;
    start.store( true, std::memory_order_release );

    while( !done.load( std::memory_order_acquire ) )
    {
        if( snapshot.consume( value ) )
        {
            valid    = valid && coherent( value ) && value.m_sequence > previous;
            previous = value.m_sequence;
            ++received;
        }

        std::this_thread::sleep_for( std::chrono::microseconds( 20 ) );
    }

    producer.join();

    if( snapshot.consume( value ) )
    {
        valid    = valid && coherent( value ) && value.m_sequence > previous;
        previous = value.m_sequence;
        ++received;
    }

    CHECK( valid );
    CHECK( received > 0 );
    CHECK( received <= publications );
    CHECK( previous == publications );
    CHECK_FALSE( snapshot.consume( value ) );
    CHECK( value.m_sequence == publications );
    CHECK( coherent( value ) );
}

} // namespace latestSnapshotTest
} // namespace libXWCTest
