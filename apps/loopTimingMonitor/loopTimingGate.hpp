/** \file loopTimingGate.hpp
 * \brief Subscription-state qualification for loop timing observations.
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

/// Require affirmative camera, DM and loop states without hiding intervening transitions.
/** Received states remain valid until another subscription update, explicit
 * invalidation or reset. The caller serializes mutable access. Generation
 * changes invalidate queued observations even when an off/on transition occurs
 * between polls.
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

    /// Retain an affirmative or negative remote state until explicitly changed or invalidated.
    void update( Source source /**< [in] Camera, DM or loop subscription. */,
                 bool   value /**< [in] Whether the received state satisfies its prerequisite. */ );

    /// Revoke a subscription after explicitly detected missing or malformed state information.
    void invalidate( Source source /**< [in] Subscription whose state is no longer known. */ );

    /// Return whether all last known subscription states satisfy their prerequisites.
    bool enabled() const;

    /// Return the transition generation for tagging and rejecting queued observations.
    uint64_t generation() const;

    /// Forget all subscriptions and invalidate observations from the preceding generation.
    void reset();

  private:
    /// One retained subscription state, including whether any usable update has been received.
    struct State
    {
        bool known{ false }; ///< A state has been received and has not been explicitly invalidated.

        bool value{ false }; ///< The most recent state satisfies its prerequisite.
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

inline void Gate::update( Source source, bool value )
{
    const auto index = static_cast<size_t>( source );

    if( index >= m_states.size() )
    {
        return;
    }

    auto &state = m_states[index];

    if( !state.known || state.value != value )
    {
        advance();
    }

    state = { true, value };
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

inline bool Gate::enabled() const
{
    bool ready = true;

    for( const auto &state : m_states )
    {
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
