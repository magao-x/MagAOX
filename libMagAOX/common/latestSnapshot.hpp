/** \file latestSnapshot.hpp
 * \brief Ownership-safe latest-value handoff for one producer and one consumer.
 */
#ifndef common_latestSnapshot_hpp
#define common_latestSnapshot_hpp

#include <array>
#include <atomic>
#include <cstdint>
#include <type_traits>

namespace MagAOX
{

/// Publish current fixed-size state without waiting for or overwriting the reader.
/** Exactly one serialized producer calls publish and one serialized consumer calls
 * consume. Three slots remain exclusively producer-owned, consumer-owned or in
 * the atomic handoff. Unread publications coalesce into the newest value; this
 * is a state snapshot, not an event queue. A successful consume copies the latest
 * value at its ownership exchange, leaving the caller's copy independent of
 * later publications. Neither operation allocates, takes a mutex or retries.
 */
template <class valueT>
class latestSnapshot
{
    static_assert( std::is_trivially_copyable_v<valueT>, "Snapshot values must be trivially copyable" );
    static_assert( std::is_nothrow_copy_assignable_v<valueT>, "Snapshot values must permit nonthrowing assignment" );
    static_assert( std::atomic<uint8_t>::is_always_lock_free, "Snapshot ownership exchange must be lock-free" );

    static constexpr uint8_t dirty{ 4 };     ///< Marks a handoff value not yet acquired by the consumer.
    static constexpr uint8_t indexMask{ 3 }; ///< Extracts one of the three fixed slot indices.

    std::array<valueT, 3> m_slots{};     ///< Inline payload storage; only its current owner accesses each slot.
    uint8_t               m_writer{ 0 }; ///< Producer-owned slot, never exposed until its copy is complete.
    uint8_t               m_reader{ 1 }; ///< Consumer-owned slot, retained throughout each output copy.
    std::atomic<uint8_t>  m_middle{ 2 }; ///< Exchange-owned slot index and unread-publication flag.

  public:
    /// Start with distinct owners and no published value; construct before either owner starts.
    latestSnapshot() = default;

    /// Replace the unread state with this value and retain a slot for the next publication.
    void publish( const valueT &value /**< [in] Producer-owned value copied before publication. */ ) noexcept;

    /// Copy the newest available state, leaving the output unchanged when none is new.
    bool consume( valueT &value /**< [out] Consumer-owned copy on success. */ ) noexcept;
};

template <class valueT>
inline void latestSnapshot<valueT>::publish( const valueT &value ) noexcept
{
    m_slots[m_writer] = value;
    // Release the completed payload; acquire any slot released by the consumer's preceding copy.
    m_writer = m_middle.exchange( uint8_t( m_writer | dirty ), std::memory_order_acq_rel ) & indexMask;
}

template <class valueT>
inline bool latestSnapshot<valueT>::consume( valueT &value ) noexcept
{
    if( !( m_middle.load( std::memory_order_acquire ) & dirty ) )
    {
        return false;
    }

    // Release the old reader slot; acquire the published slot before touching its ordinary payload.
    m_reader = m_middle.exchange( m_reader, std::memory_order_acq_rel ) & indexMask;
    value    = m_slots[m_reader];
    return true;
}

} // namespace MagAOX
#endif
