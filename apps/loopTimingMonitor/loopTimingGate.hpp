/** \file loopTimingGate.hpp
 * \brief Fresh remote-state qualification for loop timing observations.
 */
#ifndef loopTimingMonitor_loopTimingGate_hpp
#define loopTimingMonitor_loopTimingGate_hpp

#include <array>
#include <cstddef>
#include <cstdint>

namespace MagAOX
{
namespace app
{
namespace loopTiming
{

/// Require fresh affirmative camera, DM and loop states without hiding intervening transitions.
/** The caller serializes updates and expiry. Generation changes invalidate
 * queued observations even when an off/on transition occurs between polls.
 */
class Gate
{
  public:
    /// Independently observed prerequisites for accepting timing samples.
    enum class Source : size_t
    {
        camera,
        dm,
        loop
    };

    /// Record an affirmative or negative remote state at its local receipt time.
    void update( Source  source /**< [in] Camera, DM or loop subscription. */,
                 bool    value /**< [in] Whether the received state satisfies its prerequisite. */,
                 int64_t nowNs /**< [in] Positive monotonic receipt timestamp. */ );

    /// Revoke a subscription after missing or malformed state information.
    void invalidate( Source source /**< [in] Subscription whose state is no longer known. */ );

    /// Expire old or regressing observations and return whether all prerequisites hold.
    bool enabled( int64_t nowNs /**< [in] Current positive monotonic time. */,
                  int64_t timeoutNs /**< [in] Strict positive maximum receipt age. */ );

    /// Return the transition generation for tagging and rejecting queued observations.
    uint64_t generation() const;

    /// Forget all subscriptions and invalidate observations from the preceding generation.
    void reset();

  private:
    /// One locally timestamped subscription, independent of its last boolean value.
    struct State
    {
        bool known{ false }; ///< A well-formed state has been received and has not expired.

        bool value{ false }; ///< The most recent state satisfies its prerequisite.

        int64_t receivedNs{ 0 }; ///< Local monotonic receipt time, never a remote clock.
    };

    std::array<State, 3> m_states{}; ///< Caller-serialized camera, DM and loop observations.

    uint64_t m_generation{ 1 }; ///< Changes whenever prior observation eligibility is invalidated.

    /// Advance the nonzero transition generation.
    void advance();
};

inline void Gate::advance()
{
    ++m_generation;

    if( m_generation == 0 )
    {
        ++m_generation;
    }
}

inline void Gate::update( Source source, bool value, int64_t nowNs )
{
    const auto index = static_cast<size_t>( source );

    if( index >= m_states.size() )
    {
        return;
    }

    auto &state = m_states[index];

    if( nowNs <= 0 || ( state.known && nowNs < state.receivedNs ) )
    {
        invalidate( source );
        return;
    }

    if( !state.known || state.value != value )
    {
        advance();
    }

    state = { true, value, nowNs };
}

inline void Gate::invalidate( Source source )
{
    const auto index = static_cast<size_t>( source );

    if( index >= m_states.size() )
    {
        return;
    }

    m_states[index] = {};
    advance();
}

inline bool Gate::enabled( int64_t nowNs, int64_t timeoutNs )
{
    bool ready = true;

    for( auto &state : m_states )
    {
        if( state.known &&
            ( nowNs <= 0 || timeoutNs <= 0 || nowNs < state.receivedNs || nowNs - state.receivedNs >= timeoutNs ) )
        {
            state = {};
            advance();
        }

        ready = ready && state.known && state.value;
    }

    return ready;
}

inline uint64_t Gate::generation() const
{
    return m_generation;
}

inline void Gate::reset()
{
    m_states = {};
    advance();
}

} // namespace loopTiming
} // namespace app
} // namespace MagAOX

#endif // loopTimingMonitor_loopTimingGate_hpp
