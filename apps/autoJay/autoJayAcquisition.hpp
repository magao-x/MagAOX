/** \file autoJayAcquisition.hpp
 * \brief Bounded acquisition of an unpoked image and two separate DM responses.
 * \ingroup autoJay_files
 */
#ifndef autoJayAcquisition_hpp
#define autoJayAcquisition_hpp

#include "autoJayTypes.hpp"

namespace MagAOX
{
namespace app
{
namespace autoJayDetail
{

/// Hardware-independent sequence; the caller owns synchronization and channel cleanup.
class acquisition
{
  public:
    /// Ordered commands, including acknowledged removal before the other DM is poked.
    enum class step
    {
        idle,
        baseline,
        ncpc,
        removeNcpc,
        tweeter,
        cleanup,
        complete,
        failed
    };

  private:
    step                    m_step{ step::idle };      ///< Current command/acquisition phase.
    bool                    m_commandPending{ false }; ///< A channel write must precede trigger/frame acceptance.
    bool                    m_triggered{ false };      ///< Required combined streams have advanced after the write.
    std::array<uint64_t, 2> m_triggers{};              ///< Combined counters preceding the channel write.
    std::array<uint64_t, 2> m_latestTriggers{};        ///< Latest observed counters, including accumulation time.
    uint64_t                m_lastFrame{ 0 };          ///< Last observed source frame, including discarded frames.
    bool                    m_haveFrame{ false };      ///< Distinguishes the first frame from a counter reset.
    unsigned                m_frames{ 5 };             ///< Required distinct frames per stack.
    unsigned                m_settleFrames{ 2 };       ///< Required distinct settling frames.
    unsigned                m_settled{ 0 };            ///< Distinct settling frames seen in this phase.
    double                  m_delay{ 0.01 };           ///< Minimum delay after an observed combined command.
    double                  m_timeout{ 5 };            ///< Deadline interval for each phase.
    double m_saturation{ std::numeric_limits<double>::max() }; ///< Reject saturated frames before averaging.
    double m_deadline{ 0 };                                    ///< Monotonic phase deadline.
    double m_acceptAfter{ 0 };                                 ///< Earliest accepted frame reception time.
    std::array<frameStack, 3> m_stacks;                        ///< Baseline, NCPC, and tweeter statistics.
    std::string               m_error;                         ///< First failure reason.

    /// Enter the next phase and require a fresh channel command.
    void next( step value, /**< [in] New phase. */ double now /**< [in] Monotonic seconds. */ );

  public:
    /// Initialize a new transaction after the caller verifies channel ownership.
    bool start( const calibration &cal, /**< [in] Frozen acquisition settings. */
                double             now, /**< [in] Monotonic seconds. */
                double saturation = std::numeric_limits<double>::max() /**< [in] Exclusive raw-pixel upper bound. */ );

    /// Return the current phase.
    step state() const;

    /// Report whether a phase still requires its channel write.
    bool commandPending() const;

    /// Return per-channel pattern scales for the current command.
    std::array<float, 2> command() const;

    /// Mark a successful channel write using counters sampled before the write.
    void commanded( const std::array<uint64_t, 2> &counters, /**< [in] Pre-write combined counters. */
                    double                         now /**< [in] Monotonic seconds. */ );

    /// Advance trigger waiting and enforce the phase deadline.
    void poll( const std::array<uint64_t, 2> &counters, /**< [in] Current combined counters. */
               double                         now,      /**< [in] Monotonic seconds. */
               bool applied = true /**< [in] Both expected channel contributions are observed. */ );

    /// Accumulate one coherent, distinct camera snapshot after settling.
    void frame( const imageT &image,   /**< [in] Copied camera pixels. */
                uint64_t      counter, /**< [in] Source frame identity. */
                double        now /**< [in] Monotonic reception time. */ );

    /// Invalidate the sequence; the caller must clear both owned channels.
    void fail( const std::string &reason /**< [in] Failure description. */ );

    /// Return the first failure description.
    const std::string &error() const;

    /// Return completed statistics, valid only when the sequence is complete.
    const std::array<frameStack, 3> &stacks() const;
};

inline void acquisition::next( step value, double now )
{
    m_step           = value;
    m_commandPending = true;
    m_triggered      = false;
    m_settled        = 0;
    m_deadline       = now + m_timeout;
}

inline bool acquisition::start( const calibration &cal, double now, double saturation )
{
    if( cal.frames < 2 || cal.frames > 10000 || cal.settleFrames > 10000 || !mx::math::isFinite( cal.dmSleep ) ||
        cal.dmSleep < 0 || !mx::math::isFinite( cal.phaseTimeout ) || cal.phaseTimeout <= cal.dmSleep ||
        !mx::math::isFinite( saturation ) || saturation <= 0 )
    {
        fail( "invalid acquisition timing or frame counts" );
        return false;
    }
    m_frames       = cal.frames;
    m_settleFrames = cal.settleFrames;
    m_delay        = cal.dmSleep;
    m_timeout      = cal.phaseTimeout;
    m_saturation   = saturation;
    m_stacks       = {};
    m_haveFrame    = false;
    m_error.clear();
    next( step::baseline, now );
    return true;
}

inline acquisition::step acquisition::state() const
{
    return m_step;
}
inline bool acquisition::commandPending() const
{
    return m_commandPending;
}
inline std::array<float, 2> acquisition::command() const
{
    return { m_step == step::ncpc ? 1.0F : 0.0F, m_step == step::tweeter ? 1.0F : 0.0F };
}

inline void acquisition::commanded( const std::array<uint64_t, 2> &counters, double now )
{
    if( !m_commandPending )
        return;
    m_triggers       = counters;
    m_latestTriggers = counters;
    m_commandPending = false;
    m_deadline       = now + m_timeout;
}

inline void acquisition::poll( const std::array<uint64_t, 2> &counters, double now, bool applied )
{
    if( m_step == step::idle || m_step == step::failed || m_step == step::complete )
        return;
    if( now > m_deadline )
    {
        fail( "acquisition phase timed out" );
        return;
    }
    if( m_commandPending )
        return;
    if( counters[0] < m_latestTriggers[0] || counters[1] < m_latestTriggers[1] )
    {
        fail( "combined DM counter reset" );
        return;
    }
    m_latestTriggers = counters;
    if( m_triggered )
        return;
    // Both channels are written on every transition, so both combined streams must respond.
    if( applied && counters[0] > m_triggers[0] && counters[1] > m_triggers[1] )
    {
        m_triggered   = true;
        m_acceptAfter = now + m_delay;
    }
}

inline void acquisition::frame( const imageT &image, uint64_t counter, double now )
{
    if( m_step == step::idle || m_step == step::failed || m_step == step::complete )
        return;
    if( m_haveFrame && counter < m_lastFrame )
    {
        fail( "camera counter reset" );
        return;
    }
    if( m_haveFrame && counter == m_lastFrame )
        return;
    m_lastFrame = counter;
    m_haveFrame = true;
    if( m_commandPending || !m_triggered || now < m_acceptAfter )
        return;
    if( now > m_deadline )
    {
        fail( "camera phase timed out" );
        return;
    }
    if( m_settled < m_settleFrames )
    {
        ++m_settled;
        return;
    }
    if( m_step == step::removeNcpc )
    {
        next( step::tweeter, now );
        return;
    }
    if( m_step == step::cleanup )
    {
        m_step = step::complete;
        return;
    }
    const size_t index = m_step == step::baseline ? 0 : ( m_step == step::ncpc ? 1 : 2 );
    auto        &stack = m_stacks[index];
    if( image.size() == 0 )
    {
        fail( "empty camera image" );
        return;
    }
    for( Eigen::Index n = 0; n < image.size(); ++n )
    {
        if( !mx::math::isFinite( image.data()[n] ) )
        {
            fail( "nonfinite camera image" );
            return;
        }
        if( image.data()[n] >= m_saturation )
        {
            fail( "saturated camera frame" );
            return;
        }
    }
    if( stack.count == 0 )
    {
        if( index && ( image.rows() != m_stacks[0].mean.rows() || image.cols() != m_stacks[0].mean.cols() ) )
        {
            fail( "camera dimensions changed during acquisition" );
            return;
        }
        stack.mean         = image;
        stack.variance     = imageT::Zero( image.rows(), image.cols() );
        stack.firstCounter = counter;
    }
    else
    {
        if( stack.mean.rows() != image.rows() || stack.mean.cols() != image.cols() )
        {
            fail( "camera dimensions changed during stack" );
            return;
        }
        imageT delta = image - stack.mean;
        stack.mean += delta / static_cast<float>( stack.count + 1 );
        stack.variance += delta * ( image - stack.mean );
    }
    ++stack.count;
    stack.lastCounter = counter;
    if( stack.count == m_frames )
    {
        stack.variance /= static_cast<float>( stack.count * ( stack.count - 1 ) );
        next( index == 0 ? step::ncpc : ( index == 1 ? step::removeNcpc : step::cleanup ), now );
    }
}

inline void acquisition::fail( const std::string &reason )
{
    if( m_step != step::failed )
        m_error = reason;
    m_step           = step::failed;
    m_commandPending = false;
}
inline const std::string &acquisition::error() const
{
    return m_error;
}
inline const std::array<frameStack, 3> &acquisition::stacks() const
{
    return m_stacks;
}

} // namespace autoJayDetail
} // namespace app
} // namespace MagAOX
#endif // autoJayAcquisition_hpp
