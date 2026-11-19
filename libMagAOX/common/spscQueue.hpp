/** \file spscQueue.hpp
 * \brief Fixed-storage single-producer/single-consumer handoff with nonblocking loss accounting.
 */
#ifndef common_spscQueue_hpp
#define common_spscQueue_hpp

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace MagAOX
{
/// Transfer fixed-size values without overwriting a slot still owned by the reader.
/** Configure only while both owners are stopped. Exactly one producer calls push;
 * exactly one consumer calls pop. A producer that catches the consumer drops the
 * new value. Atomic publication and release of each slot protect ordinary payload.
 */
template <class valueT>
class spscQueue
{
    static_assert( std::is_trivially_copyable_v<valueT>, "SPSC values must be trivially copyable" );
    static_assert( std::atomic<uint64_t>::is_always_lock_free, "SPSC positions must be lock-free" );

    std::vector<valueT>   m_slots;        ///< Storage allocated and initialized before either owner starts.
    std::atomic<uint64_t> m_head{ 0 };    ///< Next producer sequence, released after payload copy.
    std::atomic<uint64_t> m_tail{ 0 };    ///< Next consumer sequence, released after payload copy.
    std::atomic<uint64_t> m_dropped{ 0 }; ///< Producer-only loss counter, readable by the consumer.

  public:
    /// Allocate initialized fixed storage and clear positions while both owners are stopped.
    void configure( size_t capacity /**< [in] Maximum unread values; zero disables storage. */ );

    /// Publish one value or count a drop without waiting.
    bool push( const valueT &value /**< [in] Producer-owned value to copy. */ );

    /// Copy and release one published slot; false means no value was ready.
    bool pop( valueT &value /**< [out] Consumer-owned copy. */ );

    /// Return immutable configured capacity.
    size_t capacity() const;

    /// Return the bounded observed unread population; capture once to bound a consumer drain.
    size_t queued() const;

    /// Return cumulative losses since configure.
    uint64_t dropped() const;
};

template <class valueT>
inline void spscQueue<valueT>::configure( size_t capacity )
{
    m_slots.assign( capacity, valueT{} );
    m_head.store( 0, std::memory_order_relaxed );
    m_tail.store( 0, std::memory_order_relaxed );
    m_dropped.store( 0, std::memory_order_relaxed );
}

template <class valueT>
inline bool spscQueue<valueT>::push( const valueT &value )
{
    const uint64_t head = m_head.load( std::memory_order_relaxed );
    const uint64_t tail = m_tail.load( std::memory_order_acquire );

    if( head - tail == m_slots.size() )
    {
        m_dropped.fetch_add( 1, std::memory_order_relaxed );
        return false;
    }

    m_slots[head % m_slots.size()] = value;
    m_head.store( head + 1, std::memory_order_release );
    return true;
}

template <class valueT>
inline bool spscQueue<valueT>::pop( valueT &value )
{
    const uint64_t tail = m_tail.load( std::memory_order_relaxed );
    const uint64_t head = m_head.load( std::memory_order_acquire );

    if( tail == head )
    {
        return false;
    }

    value = m_slots[tail % m_slots.size()];
    m_tail.store( tail + 1, std::memory_order_release );
    return true;
}

template <class valueT>
inline size_t spscQueue<valueT>::capacity() const
{
    return m_slots.size();
}

template <class valueT>
inline size_t spscQueue<valueT>::queued() const
{
    const uint64_t head = m_head.load( std::memory_order_acquire );
    const uint64_t tail = m_tail.load( std::memory_order_acquire );
    return static_cast<size_t>( std::min<uint64_t>( head - tail, m_slots.size() ) );
}

template <class valueT>
inline uint64_t spscQueue<valueT>::dropped() const
{
    return m_dropped.load( std::memory_order_relaxed );
}
} // namespace MagAOX
#endif
