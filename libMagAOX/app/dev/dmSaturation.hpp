/** \file dmSaturation.hpp
 * \brief Preallocated saturation payload accumulation between command publication and reporting.
 */
#ifndef dev_dmSaturation_hpp
#define dev_dmSaturation_hpp

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Preserve every successful command's saturation mask while notification delivery is delayed.
/** Configure only while both owners are stopped. Each successful add contributes
 * one sample and its per-actuator flags. A drain transfers all pending samples
 * into caller-owned window totals; repeated notifications cannot duplicate them.
 * The bounded storage is allocated only during configure, and neither owner
 * holds the mutex while publishing images or computing reporting statistics.
 */
class dmSaturation
{
  protected:
    mutable std::mutex m_mutex; ///< Protects pending counts during the short add/drain operations.

    std::vector<uint64_t> m_pending; ///< Preallocated pending per-actuator sums, shared only under the mutex.

    uint64_t m_samples{ 0 }; ///< Number of command masks represented by the pending sums.

  public:
    /// Allocate zeroed per-actuator storage and discard old samples while both owners are stopped.
    void configure( size_t pixels /**< [in] Fixed mask size; zero disables accumulation. */ );

    /// Add one complete binary mask, returning false without mutation on invalid input or overflow.
    bool add( const uint8_t *mask /**< [in] Immutable per-actuator flags, each zero or one. */,
              size_t         count /**< [in] Number of flags; must match the configured nonzero size. */ );

    /// Add and clear all pending sums; return the sample count, or zero without mutation on invalid input/overflow.
    uint64_t drain( uint64_t *totals /**< [in,out] Caller-owned window totals, retained between drains. */,
                    size_t    count /**< [in] Number of totals; must match the configured nonzero size. */ );

    /// Discard pending masks under the ownership mutex without reallocating configured storage.
    void clear() noexcept;

    /// Return whether undrained masks remain; this snapshot does not claim their ownership.
    bool pending() const;
};

inline void dmSaturation::configure( size_t pixels )
{
    m_pending.assign( pixels, 0 );
    m_samples = 0;
}

inline bool dmSaturation::add( const uint8_t *mask, size_t count )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    constexpr auto              limit = std::numeric_limits<uint64_t>::max();

    if( mask == nullptr || count == 0 || count != m_pending.size() || m_samples == limit )
    {
        return false;
    }

    for( size_t index = 0; index < count; ++index )
    {
        if( mask[index] > 1 || m_pending[index] > limit - mask[index] )
        {
            return false;
        }
    }

    for( size_t index = 0; index < count; ++index )
    {
        m_pending[index] += mask[index];
    }

    ++m_samples;
    return true;
}

inline uint64_t dmSaturation::drain( uint64_t *totals, size_t count )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    constexpr auto              limit = std::numeric_limits<uint64_t>::max();

    if( totals == nullptr || count == 0 || count != m_pending.size() || m_samples == 0 )
    {
        return 0;
    }

    for( size_t index = 0; index < count; ++index )
    {
        if( totals[index] > limit - m_pending[index] )
        {
            return 0;
        }
    }

    for( size_t index = 0; index < count; ++index )
    {
        totals[index] += m_pending[index];
        m_pending[index] = 0;
    }

    const uint64_t samples = m_samples;
    m_samples              = 0;
    return samples;
}

inline bool dmSaturation::pending() const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_samples != 0;
}

inline void dmSaturation::clear() noexcept
{
    std::lock_guard<std::mutex> lock( m_mutex );

    for( auto &value : m_pending )
    {
        value = 0;
    }

    m_samples = 0;
}

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
