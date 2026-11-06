/** \file training_profile.hpp
 * \brief Opt-in offline phase counters; all hooks disappear from ordinary application/test builds.
 */
#ifndef PCTRAININGPROFILE_HPP
#define PCTRAININGPROFILE_HPP

#ifdef DDSPC_TRAINING_PROFILE
    #include <array>
    #include <cstdint>
    #include <ctime>
namespace DDSPCProfile
{
/// Inclusive phase labels; parent/child times must not be added as independent costs.
enum Phase
{
    History,
    Weights,
    Covariance,
    QrFactor,
    QrSolve,
    Rank,
    Reference,
    Diagnostics,
    Snapshot,
    Queue,
    Processing,
    Count
};
/// Accumulate nanoseconds and calls without allocating.
struct Sample
{
    std::uint64_t m_wallNs{ 0 }, m_cpuNs{ 0 }; ///< Inclusive elapsed and thread CPU nanoseconds.
    std::uint64_t m_calls{ 0 };                ///< Number of measured phase invocations.
};
inline thread_local bool                      enabled = false; ///< Only the selected profiling scope is observed.
inline thread_local std::array<Sample, Count> samples{}; ///< Per-thread fixed counters, never shared between owners.
/// Read a supported monotonic/CPU clock; zero explicitly indicates unsupported profiling.
inline std::uint64_t stamp( clockid_t clock /**< [in] Clock domain. */ )
{
    timespec t{};
    return clock_gettime( clock, &t ) ? 0 : std::uint64_t( t.tv_sec ) * 1000000000 + t.tv_nsec;
}
/// A stack-only phase timer; explicit finish permits queue prologue timing before numerical work.
class Scope
{
    Phase         m_phase;                 ///< Destination phase.
    std::uint64_t m_wall{ 0 }, m_cpu{ 0 }; ///< Start clocks, zero when disabled/finished.
  public:
    /// Begin a phase only when profiling is enabled.
    explicit Scope( Phase phase /**< [in] Inclusive phase label. */ );
    /// Finish once on normal/exception/early-return scope exit.
    ~Scope();
    /// Finish before leaving a larger scope; repeated calls are harmless.
    void finish();
};
inline Scope::Scope( Phase phase ) : m_phase( phase )
{
    if( enabled )
    {
        m_cpu  = stamp( CLOCK_THREAD_CPUTIME_ID );
        m_wall = stamp( CLOCK_MONOTONIC );
    }
}
inline Scope::~Scope()
{
    finish();
}
inline void Scope::finish()
{
    if( !m_wall )
        return;
    auto &s = samples[m_phase];
    s.m_wallNs += stamp( CLOCK_MONOTONIC ) - m_wall;
    s.m_cpuNs += stamp( CLOCK_THREAD_CPUTIME_ID ) - m_cpu;
    ++s.m_calls;
    m_wall = 0;
}
} //namespace DDSPCProfile
    #define DDSPC_PROFILE_BEGIN( phase, name ) DDSPCProfile::Scope name( DDSPCProfile::phase )
    #define DDSPC_PROFILE_END( name ) name.finish()
#else
    #define DDSPC_PROFILE_BEGIN( phase, name )
    #define DDSPC_PROFILE_END( name )
#endif
#define DDSPC_PROFILE_SCOPE( phase ) DDSPC_PROFILE_BEGIN( phase, profile##phase )
#endif
