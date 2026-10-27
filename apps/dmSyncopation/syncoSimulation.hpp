/** \file syncoSimulation.hpp
 * \brief Bounded, event-driven prediction of syncopation policy without stream access or DM writes.
 */
#ifndef dmSyncopation_syncoSimulation_hpp
#define dmSyncopation_syncoSimulation_hpp

#include <array>
#include <limits>

#include <mx/math/floatUtils.hpp>

#include "syncoSchedule.hpp"
#include "../../libMagAOX/common/dmSyncoSimulation.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Pure conditional scheduler; the caller serializes timestamp-ordered event delivery.
class Simulation
{
  public:
    /// Startup-only model settings; these bounds do not qualify instrument safety.
    struct Config
    {
        Bounds   bounds;                  ///< Assumed copy, reader and next-main bounds.
        int      steps{ 2 };              ///< Fixed requested total steps, including passive zero.
        unsigned window{ 32 };            ///< Retained publication intervals and phase observations.
        int64_t  nominalPeriod{ 500000 }; ///< Initial period in monotonic nanoseconds.
        int64_t  minInterval{ 250000 };   ///< Smallest eligible publication interval.
        int64_t  maxInterval{ 2000000 };  ///< Largest eligible publication interval.
        double   phaseGain{ 0.1 };        ///< Bounded correction toward the robust phase candidate.
        int64_t  phaseSlew{ 1000 };       ///< Maximum correction per newly committed cycle.
        int64_t  cadenceMinPeriod{ 0 };   ///< Minimum whole-cycle period admitted by the nominal cadence rating.
        int64_t  timeout{ 4000000 };      ///< Maximum primary silence or eligible-period-sample age before expiry.
    };

    /// Validate configuration and begin a fresh simulation with empty estimators.
    bool configure( const Config &config /**< [in] fixed startup model */ );

    /// Begin a new session with empty model history, commitments and conditional counters.
    void reset();

    /// Observe one coherent main; zero publication interval means unavailable, with no clock fallback.
    void
    primary( uint64_t counter /**< [in] genuine primary counter */,
             int64_t  observed /**< [in] coherent same-host monotonic observation */,
             int64_t  publicationInterval = 0 /**< [in] qualified producer-clock interval, ns */,
             const MagAOX::dmSyncoSimulation::Observation &context = {} /**< [in] optional raw monitor context */ );

    /// Observe coherent combined-output linkage; false coherence never supplies an acknowledgement.
    void
    combined( uint64_t counter /**< [in] actual combined counter */,
              uint64_t primaryCounter /**< [in] primary counter named by its trace */,
              int64_t  observed /**< [in] same-host monotonic observation */,
              bool     coherent = true /**< [in] metadata and trace association are usable */,
              const MagAOX::dmSyncoSimulation::Observation &context = {} /**< [in] optional raw monitor context */ );

    /// Evaluate committed slots through a monotonic time without synthesizing main events.
    void advance( int64_t now /**< [in] same-host monotonic evaluation time */ );

    /// Return the bounded snapshot; callers copy it under their own serialization.
    const MagAOX::dmSyncoSimulation::Summary &summary() const;

    /// Remove the oldest retained overlap; caller serialization must match primary/combined updates.
    bool popOverlap( MagAOX::dmSyncoSimulation::OverlapEvent &event /**< [out] oldest event on success */ );

    /// Return process-lifetime FIFO accounting without consuming events; logFailed is app-owned.
    MagAOX::dmSyncoSimulation::TraceStatus traceStatus() const;

    /// Remove the oldest selected estimator update without changing model history or decisions.
    bool popEstimator( MagAOX::dmSyncoSimulation::EstimatorSample &sample /**< [out] oldest update on success */ );

    /// Return process-lifetime triggered estimator coverage without consuming records.
    MagAOX::dmSyncoSimulation::EstimatorStatus estimatorStatus() const;

  private:
    /// One primary phase observation; counters label cycles independently of time proximity.
    struct Point
    {
        uint64_t counter{ 0 };  ///< Primary command identity.
        int64_t  observed{ 0 }; ///< Coherent monotonic observation.
    };

    /// An advertised cycle whose targets are never recalculated.
    struct Plan
    {
        uint64_t                      counter{ 0 };     ///< Genuine primary counter expected for this cycle.
        uint64_t                      estimatorId{ 0 }; ///< Accepted-primary update that committed this plan.
        int64_t                       grid{ 0 };        ///< Immutable main prediction, ns.
        int64_t                       period{ 0 };      ///< Period used when committing this cycle.
        int64_t                       main{ 0 };        ///< Actual associated primary observation, or zero.
        int64_t                       output{ 0 };      ///< Coherent combined observation, or zero.
        std::array<int64_t, maxSteps> starts{};         ///< Fixed modeled copy starts.
        std::array<int64_t, maxSteps> targets{};        ///< Fixed modeled copy completion/trigger targets.
        std::array<bool, maxSteps>    done{};           ///< Each occurrence contributes to outcomes at most once.
    };

    /// Coherent output retained until its primary monitor catches up.
    struct Link
    {
        uint64_t counter{ 0 };   ///< Referenced primary counter.
        int64_t  observed{ 0 };  ///< Monotonic combined observation.
        bool     usable{ true }; ///< Exact association remains unambiguous.
    };

    std::array<Link, 4>                m_links{};        ///< Bounded future-output association buffer.
    size_t                             m_linkCount{ 0 }; ///< Initialized future-output links.
    Config                             m_config;         ///< Immutable configuration after configure.
    MagAOX::dmSyncoSimulation::Summary m_summary;        ///< Current-session conditional diagnostics and grid state.
    std::array<int64_t, 256>           m_intervals{};    ///< Fixed publication-interval ring.
    std::array<Point, 256>             m_points{};       ///< Fixed counter-labelled phase ring.
    std::array<int64_t, 256>           m_scratch{};      ///< Median workspace; no event-path allocation.
    size_t                             m_intervalCount{ 0 }; ///< Number of initialized interval entries.
    size_t                             m_intervalNext{ 0 };  ///< Next interval ring replacement.
    size_t                             m_pointCount{ 0 };    ///< Number of initialized phase entries.
    size_t                             m_pointNext{ 0 };     ///< Next phase ring replacement.
    std::array<Plan, 4>                m_plans{}; ///< Bounded advertised horizon, normally current and next cycle.
    size_t                             m_planCount{ 0 };        ///< Number of committed plans.
    bool                               m_haveMain{ false };     ///< Main identity baseline exists.
    bool                               m_haveCombined{ false }; ///< Combined identity baseline exists.
    uint64_t                           m_combinedCounter{ 0 };  ///< Last actual combined counter.
    bool                               m_mainOutput{ false };   ///< Latest actual main has a coherent combination.
    int64_t                            m_outputTime{ 0 };       ///< Latest main's combined observation.
    int64_t                            m_virtualHold{ 0 };      ///< Last modeled active command's reader lifetime.
    int64_t  m_periodTime{ 0 };      ///< Most recent eligible period sample, on the observation clock.
    int64_t  m_copyBegin{ 0 };       ///< Last modeled copy's start, retained for retrospective overlap detection.
    int64_t  m_copyEnd{ 0 };         ///< Last modeled copy's completion/trigger time.
    int      m_copyStep{ 0 };        ///< Step associated with the last modeled copy/reader hold.
    uint64_t m_copyCounter{ 0 };     ///< Genuine counter labelling the last modeled copy's cycle.
    int64_t  m_copyGrid{ 0 };        ///< Frozen prediction for the last modeled copy's cycle.
    int64_t  m_copyPeriod{ 0 };      ///< Period frozen with the last modeled copy's cycle.
    uint64_t m_copyEstimatorId{ 0 }; ///< Estimator update that committed the last modeled copy's plan.
    /// Fixed process-lifetime FIFO; reset/configure preserve undrained evidence.
    std::array<MagAOX::dmSyncoSimulation::OverlapEvent, MagAOX::dmSyncoSimulation::traceCapacity> m_traceEvents{};
    size_t   m_traceHead{ 0 };     ///< Oldest retained event's ring index.
    size_t   m_traceCount{ 0 };    ///< Occupied FIFO entries.
    uint64_t m_traceDetected{ 0 }; ///< Lifetime overlap evidence sequence, including dropped events.
    uint64_t m_traceDrained{ 0 };  ///< Lifetime events removed for telemetry attempts.
    uint64_t m_traceDropped{ 0 };  ///< Lifetime events discarded on FIFO saturation.
    /// Completed accepted-primary history; resets clear the history while preserving process IDs and export state.
    std::array<MagAOX::dmSyncoSimulation::EstimatorSample, MagAOX::dmSyncoSimulation::estimatorHistoryCapacity>
           m_estimatorHistory{};
    size_t m_estimatorHistoryNext{ 0 };  ///< Next pre-trigger history replacement.
    size_t m_estimatorHistoryCount{ 0 }; ///< Initialized pre-trigger history entries.
    /// Fixed export FIFO; saturation drops diagnostic records without changing the scheduler.
    std::array<MagAOX::dmSyncoSimulation::EstimatorSample, MagAOX::dmSyncoSimulation::estimatorTraceCapacity>
             m_estimatorEvents{};
    size_t   m_estimatorHead{ 0 };           ///< Oldest selected estimator record.
    size_t   m_estimatorCount{ 0 };          ///< Selected updates waiting for telemetry.
    uint64_t m_estimatorObserved{ 0 };       ///< Process-lifetime completed accepted-primary sequence.
    uint64_t m_estimatorTriggers{ 0 };       ///< Overlap observations selecting pre/post history windows.
    uint64_t m_estimatorSelected{ 0 };       ///< Unique selected updates, including export FIFO losses.
    uint64_t m_estimatorDrained{ 0 };        ///< Updates removed for ordinary-thread telemetry attempts.
    uint64_t m_estimatorDropped{ 0 };        ///< New selected updates discarded by the full FIFO.
    uint64_t m_estimatorLastSelectedId{ 0 }; ///< Last selected process ID; dropped records are never retried.
    size_t   m_estimatorPostRemaining{ 0 };  ///< Completed updates still to select after the most recent trigger.

    /// Checked prediction using relative counter arithmetic; zero signals unrepresentable time.
    static int64_t shifted( int64_t  base /**< [in] time origin */,
                            uint64_t from /**< [in] origin cycle */,
                            uint64_t to /**< [in] requested cycle */,
                            int64_t  period /**< [in] period */ );

    /// Calculate an overflow-safe even/odd median in the fixed scratch array.
    int64_t median( size_t count /**< [in] initialized scratch entries */ );

    /// Return the committed plan for a counter, or null.
    Plan *plan( uint64_t counter /**< [in] primary counter */ );

    /// Select an existing frozen prediction or extrapolate from the last committed cycle.
    int64_t prediction( uint64_t counter /**< [in] actual primary counter */ );

    /// Return the candidate monotonic phase for a future counter, or zero.
    int64_t phase( uint64_t counter /**< [in] future primary counter */ );

    /// Append one immutable plan; false means invalid/overflowing horizon.
    bool commit( uint64_t counter /**< [in] cycle identity */,
                 int64_t  grid /**< [in] frozen predicted main time */,
                 uint64_t estimatorId /**< [in] accepted-primary update committing this plan */ );

    /// Select a unique completed estimator update for export, dropping new records if the FIFO is full.
    void selectEstimator( const MagAOX::dmSyncoSimulation::EstimatorSample &sample /**< [in] completed update */ );

    /// Select retained pre-trigger history and restart the bounded subsequent-update window.
    void triggerEstimator();

    /// Finish one accepted-primary record, retain its history and select it when post-trigger coverage is active.
    void finishEstimator( MagAOX::dmSyncoSimulation::EstimatorSample &sample /**< [in,out] original update values */ );

    /// Cancel and retire plans superseded by a genuine main.
    void retire( uint64_t counter /**< [in] newest genuine primary counter */ );

    /// Count a conditional step outcome, using a member of the shared outcome type as reason.
    void skip( Plan    &entry /**< [in,out] committed cycle */,
               int      step /**< [in] step index */,
               uint64_t MagAOX::dmSyncoSimulation::Step::*reason /**< [in] skip classification */ );

    /// Evaluate one fixed copy start using actual-main associations and modeled reader holds.
    void evaluate( Plan &entry /**< [in,out] committed cycle */, int step /**< [in] step index */ );

    /// Count a main observation that could overlap the last modeled copy or active reader interval.
    void overlap( uint64_t mainCounter /**< [in] genuine primary counter supplying evidence */,
                  uint64_t combinedCounter /**< [in] current or last observed actual combined counter */,
                  int64_t  observed /**< [in] monotonic main evidence */,
                  int64_t  mainGrid /**< [in] frozen prediction for the evidence's primary counter */,
                  MagAOX::dmSyncoSimulation::OverlapOrigin      origin /**< [in] monitor supplying the evidence */,
                  const MagAOX::dmSyncoSimulation::Observation &context /**< [in] raw caller context */ );

    /// Reject current/future acknowledgements after a known output gap or incoherent observation.
    void badOutput();

    /// Process due slots through the requested time, with optional exclusion of equality.
    void through( int64_t now /**< [in] monotonic time */,
                  bool    inclusive /**< [in] whether equal-time slots are eligible */ );

    /// Refresh next committed target fields without retiming plans.
    void next();
};

inline int64_t Simulation::shifted( int64_t base, uint64_t from, uint64_t to, int64_t period )
{
    const __int128 value = __int128( base ) + ( __int128( to ) - __int128( from ) ) * period;

    if( value <= 0 || value > std::numeric_limits<int64_t>::max() )
    {
        return 0;
    }

    return int64_t( value );
}

inline bool Simulation::configure( const Config &config )
{
    if( config.steps < 1 || config.steps > maxSteps || config.window < 2 || config.window > m_intervals.size() ||
        config.minInterval <= 0 || config.maxInterval < config.minInterval || config.maxInterval > 1000000000 ||
        config.nominalPeriod < config.minInterval || config.nominalPeriod > config.maxInterval ||
        !mx::math::isFinite( config.phaseGain ) || config.phaseGain < 0 || config.phaseGain > 1 ||
        config.phaseSlew < 0 || config.phaseSlew > config.minInterval / 4 ||
        config.timeout < 2 * config.nominalPeriod || config.timeout > 1000000000000LL || config.cadenceMinPeriod < 0 ||
        config.cadenceMinPeriod > 1000000000000LL )
    {
        return false;
    }

    const auto &b = config.bounds;

    for( const auto value : { b.keepout,
                              b.guard,
                              b.delivery,
                              b.observation,
                              b.early,
                              b.lead,
                              b.visibility,
                              b.reserve,
                              b.snapshot,
                              b.copy } )
    {
        if( value < 0 || value > 1000000000 )
        {
            return false;
        }
    }

    if( b.visibility == 0 || b.reserve == 0 || b.snapshot == 0 || b.copy == 0 )
    {
        return false;
    }

    const uint64_t session = m_summary.enabled ? m_summary.session : 0;
    m_config               = config;
    m_summary              = {};
    m_summary.session      = session;
    reset();
    m_summary.resets = 0;
    return true;
}

inline void Simulation::reset()
{
    const uint64_t session = m_summary.session + 1;
    m_summary              = {};
    m_summary.session      = session;
    m_summary.resets       = 1;
    m_summary.enabled      = true;
    m_summary.valid        = false;
    m_summary.feasible     = feasible( m_config.nominalPeriod, m_config.steps, m_config.bounds ) &&
                         m_config.nominalPeriod >= m_config.cadenceMinPeriod;
    m_summary.sequence           = 0;
    m_summary.nowNs              = 0;
    m_summary.lastMainNs         = 0;
    m_summary.counter            = 0;
    m_summary.steps              = m_config.steps;
    m_summary.periodNs           = m_config.nominalPeriod;
    m_summary.gridNs             = 0;
    m_summary.phaseNs            = 0;
    m_summary.phaseMinNs         = 0;
    m_summary.phaseMaxNs         = 0;
    m_summary.phaseCount         = 0;
    m_summary.periodSamples      = 0;
    m_summary.phaseSamples       = 0;
    m_summary.nextGridNs         = 0;
    m_summary.nextActiveNs       = 0;
    m_summary.earlyReserveNs     = m_config.bounds.early;
    m_summary.deliveryNs         = m_config.bounds.delivery;
    m_summary.keepoutNs          = m_config.bounds.keepout;
    m_summary.guardNs            = m_config.bounds.guard;
    m_summary.snapshotNs         = m_config.bounds.snapshot;
    m_summary.copyNs             = m_config.bounds.copy;
    m_summary.leadNs             = m_config.bounds.lead;
    m_summary.observationNs      = m_config.bounds.observation;
    m_summary.visibilityNs       = m_config.bounds.visibility;
    m_summary.reserveNs          = m_config.bounds.reserve;
    m_summary.window             = m_config.window;
    m_summary.phaseGain          = m_config.phaseGain;
    m_summary.cadenceMinPeriodNs = m_config.cadenceMinPeriod;
    m_summary.phaseSlewNs        = m_config.phaseSlew;
    m_intervalCount = m_intervalNext = m_pointCount = m_pointNext = m_planCount = 0;
    m_linkCount                                                                 = 0;
    m_haveMain = m_haveCombined = m_mainOutput = false;
    m_outputTime = m_virtualHold = m_copyBegin = m_copyEnd = m_periodTime = 0;
    m_copyStep                                                            = 0;
    m_copyCounter                                                         = 0;
    m_copyGrid = m_copyPeriod = 0;
    m_copyEstimatorId         = 0;
    m_estimatorHistoryNext = m_estimatorHistoryCount = m_estimatorPostRemaining = 0;
}

inline int64_t Simulation::median( size_t count )
{
    auto middle = m_scratch.begin() + count / 2;
    std::nth_element( m_scratch.begin(), middle, m_scratch.begin() + count );
    const int64_t upper = *middle;

    if( count % 2 )
    {
        return upper;
    }

    const int64_t lower = *std::max_element( m_scratch.begin(), middle );
    return lower + ( upper - lower ) / 2;
}

inline Simulation::Plan *Simulation::plan( uint64_t counter )
{
    for( size_t i = 0; i < m_planCount; ++i )
    {
        if( m_plans[i].counter == counter )
        {
            return &m_plans[i];
        }
    }

    return nullptr;
}

inline int64_t Simulation::prediction( uint64_t counter )
{
    if( const auto *entry = plan( counter ) )
    {
        return entry->grid;
    }

    if( m_planCount )
    {
        const auto &last = m_plans[m_planCount - 1];
        return shifted( last.grid, last.counter, counter, last.period );
    }

    return 0;
}

inline int64_t Simulation::phase( uint64_t counter )
{
    for( size_t i = 0; i < m_pointCount; ++i )
    {
        m_scratch[i] = shifted( m_points[i].observed, m_points[i].counter, counter, m_summary.periodNs );

        if( !m_scratch[i] )
        {
            return 0;
        }
    }

    return m_pointCount ? median( m_pointCount ) : 0;
}

inline bool Simulation::commit( uint64_t counter, int64_t grid, uint64_t estimatorId )
{
    if( m_planCount == m_plans.size() || !grid || !shifted( grid, 0, 2, m_summary.periodNs ) )
    {
        ++m_summary.invalid;
        m_summary.valid = false;
        m_planCount     = 0;
        return false;
    }

    Plan entry;
    entry.counter     = counter;
    entry.estimatorId = estimatorId;
    entry.grid        = grid;
    entry.period      = m_summary.periodNs;
    entry.targets[0]  = copyCutoff( grid, m_config.bounds ) - m_config.bounds.reserve;
    entry.starts[0]   = entry.targets[0] - m_config.bounds.copy;

    for( int j = 1; j < m_config.steps; ++j )
    {
        entry.targets[j] = grid + j * entry.period / m_config.steps;
        entry.starts[j]  = entry.targets[j] - m_config.bounds.copy;
    }

    m_plans[m_planCount++] = entry;
    return true;
}

inline void Simulation::selectEstimator( const MagAOX::dmSyncoSimulation::EstimatorSample &sample )
{
    if( sample.id <= m_estimatorLastSelectedId )
    {
        return;
    }

    m_estimatorLastSelectedId = sample.id;
    ++m_estimatorSelected;

    if( m_estimatorCount == m_estimatorEvents.size() )
    {
        ++m_estimatorDropped;
        return;
    }

    m_estimatorEvents[( m_estimatorHead + m_estimatorCount ) % m_estimatorEvents.size()] = sample;
    ++m_estimatorCount;
}

inline void Simulation::triggerEstimator()
{
    ++m_estimatorTriggers;
    const size_t first =
        ( m_estimatorHistoryNext + m_estimatorHistory.size() - m_estimatorHistoryCount ) % m_estimatorHistory.size();

    for( size_t i = 0; i < m_estimatorHistoryCount; ++i )
    {
        selectEstimator( m_estimatorHistory[( first + i ) % m_estimatorHistory.size()] );
    }

    m_estimatorPostRemaining = MagAOX::dmSyncoSimulation::estimatorPostSamples;
}

inline void Simulation::finishEstimator( MagAOX::dmSyncoSimulation::EstimatorSample &sample )
{
    sample.sequence                            = m_summary.sequence;
    sample.periodSamples                       = m_summary.periodSamples;
    sample.phaseSamples                        = m_summary.phaseSamples;
    sample.periodNs                            = m_summary.periodNs;
    m_estimatorObserved                        = sample.id;
    m_estimatorHistory[m_estimatorHistoryNext] = sample;
    m_estimatorHistoryNext                     = ( m_estimatorHistoryNext + 1 ) % m_estimatorHistory.size();
    m_estimatorHistoryCount                    = std::min( m_estimatorHistoryCount + 1, m_estimatorHistory.size() );

    if( m_estimatorPostRemaining )
    {
        selectEstimator( sample );
        --m_estimatorPostRemaining;
    }
}

inline void Simulation::skip( Plan &entry, int step, uint64_t MagAOX::dmSyncoSimulation::Step::*reason )
{
    if( entry.done[step] )
    {
        return;
    }

    entry.done[step] = true;
    auto &outcome    = m_summary.outcomes[step];
    ++outcome.considered;
    ++outcome.skipped;
    ++( outcome.*reason );
}

inline void Simulation::retire( uint64_t counter )
{
    size_t retained = 0;

    for( size_t i = 0; i < m_planCount; ++i )
    {
        if( m_plans[i].counter < counter )
        {
            for( int j = 0; j < m_config.steps; ++j )
            {
                skip( m_plans[i], j, &MagAOX::dmSyncoSimulation::Step::expired );
            }
        }
        else
        {
            m_plans[retained++] = m_plans[i];
        }
    }

    m_planCount = retained;
}

inline void Simulation::evaluate( Plan &entry, int step )
{
    using Outcome        = MagAOX::dmSyncoSimulation::Step;
    const auto   &b      = m_config.bounds;
    const int64_t start  = entry.starts[step];
    const int64_t target = entry.targets[step];

    if( !feasible( entry.period, m_config.steps, b ) || entry.period < m_config.cadenceMinPeriod || start <= 0 )
    {
        skip( entry, step, &Outcome::expired );
        return;
    }

    bool knownMain    = m_summary.counter >= entry.counter;
    bool obsoleteMain = false;

    for( size_t i = 0; i < m_linkCount; ++i )
    {
        knownMain    = knownMain || ( m_links[i].counter == entry.counter && m_links[i].observed <= start );
        obsoleteMain = obsoleteMain || ( m_links[i].counter > entry.counter && m_links[i].observed <= start );
    }

    if( step == 0 && knownMain )
    {
        skip( entry, step, &Outcome::earlyMain );
        return;
    }

    if( step && obsoleteMain )
    {
        skip( entry, step, &Outcome::expired );
        return;
    }

    const bool main   = step ? entry.main > 0
                             : m_haveMain && m_summary.counter != std::numeric_limits<uint64_t>::max() &&
                                 m_summary.counter + 1 == entry.counter;
    const bool output = step ? entry.output > 0 : m_mainOutput;

    if( !main )
    {
        skip( entry, step, &Outcome::missingMain );
        return;
    }

    if( !output )
    {
        skip( entry, step, &Outcome::missingOutput );
        return;
    }

    const int64_t actual = step ? std::max( entry.main, entry.output ) : std::max( m_summary.lastMainNs, m_outputTime );
    const __int128 hold  = __int128( actual ) + b.delivery + b.keepout + b.guard;

    if( hold > start || m_virtualHold > start )
    {
        skip( entry, step, &Outcome::readerHold );
        return;
    }

    const auto    *following = plan( entry.counter + 1 );
    const int64_t  nextGrid  = following ? following->grid : shifted( entry.grid, 0, 1, entry.period );
    const __int128 readerEnd = __int128( target ) + b.delivery + b.keepout + b.guard;

    if( step && ( !nextGrid || readerEnd > __int128( nextGrid ) - b.early - b.observation ) )
    {
        skip( entry, step, &Outcome::expired );
        return;
    }

    entry.done[step] = true;
    ++m_summary.outcomes[step].considered;
    ++m_summary.outcomes[step].wouldRun;
    m_copyBegin       = start;
    m_copyEnd         = target;
    m_copyStep        = step;
    m_copyCounter     = entry.counter;
    m_copyGrid        = entry.grid;
    m_copyPeriod      = entry.period;
    m_copyEstimatorId = entry.estimatorId;

    if( step )
    {
        m_virtualHold = int64_t( readerEnd );
    }
}

inline void Simulation::next()
{
    m_summary.nextGridNs = m_summary.nextActiveNs = 0;

    for( size_t i = 0; i < m_planCount; ++i )
    {
        const auto &entry = m_plans[i];

        if( entry.grid > m_summary.nowNs && ( !m_summary.nextGridNs || entry.grid < m_summary.nextGridNs ) )
        {
            m_summary.nextGridNs = entry.grid;
        }

        for( int j = 1; j < m_config.steps; ++j )
        {
            if( !entry.done[j] && entry.targets[j] > m_summary.nowNs &&
                ( !m_summary.nextActiveNs || entry.targets[j] < m_summary.nextActiveNs ) )
            {
                m_summary.nextActiveNs = entry.targets[j];
            }
        }
    }
}

inline void Simulation::overlap( uint64_t                                      mainCounter,
                                 uint64_t                                      combinedCounter,
                                 int64_t                                       observed,
                                 int64_t                                       mainGrid,
                                 MagAOX::dmSyncoSimulation::OverlapOrigin      origin,
                                 const MagAOX::dmSyncoSimulation::Observation &context )
{
    if( m_copyBegin && ( ( observed >= m_copyBegin && observed < m_copyEnd ) ||
                         ( m_copyStep > 0 && observed >= m_copyEnd && observed < m_virtualHold ) ) )
    {
        ++m_summary.outcomes[m_copyStep].overlap;
        ++m_traceDetected;
        triggerEstimator();

        if( m_traceCount == m_traceEvents.size() )
        {
            ++m_traceDropped;
            return;
        }

        auto &event           = m_traceEvents[( m_traceHead + m_traceCount ) % m_traceEvents.size()];
        event                 = {};
        event.id              = m_traceDetected;
        event.session         = m_summary.session;
        event.mainCounter     = mainCounter;
        event.combinedCounter = combinedCounter;
        event.planCounter     = m_copyCounter;
        event.lastMainCounter = m_summary.counter;
        event.planEstimatorId = m_copyEstimatorId;
        event.lastEstimatorId = m_estimatorObserved;
        event.step            = m_copyStep;
        event.origin          = origin;
        event.window          = observed < m_copyEnd ? MagAOX::dmSyncoSimulation::OverlapWindow::copy
                                                     : MagAOX::dmSyncoSimulation::OverlapWindow::readerHold;
        event.observedNs      = observed;
        event.mainGridNs      = mainGrid;
        event.planGridNs      = m_copyGrid;
        event.periodNs        = m_copyPeriod;
        event.copyBeginNs     = m_copyBegin;
        event.copyEndNs       = m_copyEnd;
        event.readerEndNs     = m_copyStep > 0 ? m_virtualHold : 0;
        event.lastMainNs      = m_summary.lastMainNs;
        event.lastOutputNs    = m_outputTime;
        event.observation     = context;
        ++m_traceCount;
    }
}

inline void Simulation::badOutput()
{
    ++m_summary.outputGaps;
    m_mainOutput = false;

    if( auto *entry = plan( m_summary.counter ) )
    {
        entry->output = 0;
    }

    for( size_t i = 0; i < m_linkCount; ++i )
    {
        m_links[i].usable = false;
    }
}

inline void Simulation::through( int64_t now, bool inclusive )
{
    if( !m_summary.enabled || now <= 0 || now < m_summary.nowNs )
    {
        ++m_summary.invalid;
        return;
    }

    const bool staleMain   = m_haveMain && now - m_summary.lastMainNs > m_config.timeout;
    const bool stalePeriod = m_periodTime > 0 && now - m_periodTime > m_config.timeout;

    if( staleMain || stalePeriod )
    {
        for( size_t i = 0; i < m_planCount; ++i )
        {
            for( int j = 0; j < m_config.steps; ++j )
            {
                skip( m_plans[i], j, &MagAOX::dmSyncoSimulation::Step::expired );
            }
        }

        m_summary.valid  = false;
        m_summary.gridNs = 0;
        m_intervalCount = m_intervalNext = m_planCount = 0;
        m_periodTime                                   = 0;
        m_summary.periodSamples                        = 0;
        m_linkCount                                    = 0;
        m_haveCombined = m_mainOutput = false;
        m_outputTime                  = 0;

        if( staleMain )
        {
            m_pointCount = m_pointNext = 0;
            m_summary.phaseSamples     = 0;
        }
    }

    for( size_t attempts = 0; attempts < m_plans.size() * maxSteps; ++attempts )
    {
        Plan   *selected = nullptr;
        int     step     = 0;
        int64_t earliest = std::numeric_limits<int64_t>::max();

        for( size_t i = 0; i < m_planCount; ++i )
        {
            for( int j = 0; j < m_config.steps; ++j )
            {
                const int64_t start = m_plans[i].starts[j];

                if( !m_plans[i].done[j] && start < earliest && ( start < now || ( inclusive && start == now ) ) )
                {
                    selected = &m_plans[i];
                    step     = j;
                    earliest = start;
                }
            }
        }

        if( !selected )
        {
            break;
        }

        evaluate( *selected, step );
    }

    m_summary.nowNs = now;
    next();
}

inline void Simulation::advance( int64_t now )
{
    through( now, true );
}

inline void Simulation::primary( uint64_t                                      counter,
                                 int64_t                                       observed,
                                 int64_t                                       publicationInterval,
                                 const MagAOX::dmSyncoSimulation::Observation &context )
{
    if( !m_summary.enabled || observed <= 0 || observed < m_summary.nowNs )
    {
        ++m_summary.invalid;
        return;
    }

    through( observed, false );

    if( m_haveMain && counter == m_summary.counter )
    {
        ++m_summary.duplicates;
        through( observed, true );
        return;
    }

    if( m_haveMain && ( counter < m_summary.counter || observed <= m_summary.lastMainNs ) )
    {
        if( counter < m_summary.counter )
        {
            reset();
        }
        else
        {
            ++m_summary.invalid;
            return;
        }
    }

    const bool                                 sequential = m_haveMain && counter == m_summary.counter + 1;
    const bool                                 gap        = m_haveMain && !sequential;
    const int64_t                              predicted  = prediction( counter );
    MagAOX::dmSyncoSimulation::EstimatorSample estimator;
    estimator.id                    = m_estimatorObserved + 1;
    estimator.session               = m_summary.session;
    estimator.counter               = counter;
    estimator.observedNs            = observed;
    estimator.publicationIntervalNs = publicationInterval;
    estimator.sequential            = sequential;
    estimator.periodBeforeNs        = m_summary.periodNs;
    estimator.predictedNs           = predicted;
    estimator.phaseGain             = m_config.phaseGain;
    estimator.phaseSlewNs           = m_config.phaseSlew;

    if( gap )
    {
        ++m_summary.gaps;
    }

    overlap( counter,
             m_haveCombined ? m_combinedCounter : 0,
             observed,
             predicted,
             MagAOX::dmSyncoSimulation::OverlapOrigin::primary,
             context );

    retire( counter );
    m_haveMain           = true;
    m_mainOutput         = false;
    m_summary.lastMainNs = observed;
    m_summary.counter    = counter;
    ++m_summary.sequence;
    size_t retainedLinks = 0;
    bool   obsoleteLink  = false;

    for( size_t i = 0; i < m_linkCount; ++i )
    {
        obsoleteLink = obsoleteLink || m_links[i].counter < counter;
    }

    if( obsoleteLink )
    {
        badOutput();
    }

    for( size_t i = 0; i < m_linkCount; ++i )
    {
        if( m_links[i].counter == counter && m_links[i].usable )
        {
            m_mainOutput = true;
            m_outputTime = m_links[i].observed;
            ++m_summary.associations;
        }
        else if( m_links[i].counter > counter )
        {
            m_links[retainedLinks++] = m_links[i];
        }
    }

    m_linkCount = retainedLinks;

    if( sequential && publicationInterval >= m_config.minInterval && publicationInterval <= m_config.maxInterval )
    {
        m_periodTime                = observed;
        m_intervals[m_intervalNext] = publicationInterval;
        m_intervalNext              = ( m_intervalNext + 1 ) % m_config.window;
        m_intervalCount             = std::min<size_t>( m_intervalCount + 1, m_config.window );
        estimator.intervalAccepted  = true;
    }
    else if( m_summary.sequence > 1 )
    {
        ++m_summary.excluded;
    }

    m_points[m_pointNext]   = { counter, observed };
    m_pointNext             = ( m_pointNext + 1 ) % m_config.window;
    m_pointCount            = std::min<size_t>( m_pointCount + 1, m_config.window );
    m_summary.periodSamples = m_intervalCount;
    m_summary.phaseSamples  = m_pointCount;

    if( m_intervalCount == m_config.window )
    {
        std::copy_n( m_intervals.begin(), m_intervalCount, m_scratch.begin() );
        m_summary.periodNs = median( m_intervalCount );
    }

    m_summary.feasible = feasible( m_summary.periodNs, m_config.steps, m_config.bounds ) &&
                         m_summary.periodNs >= m_config.cadenceMinPeriod;

    if( predicted )
    {
        m_summary.gridNs  = predicted;
        m_summary.phaseNs = observed - predicted;
        m_summary.phaseMinNs =
            m_summary.phaseCount ? std::min( m_summary.phaseMinNs, m_summary.phaseNs ) : m_summary.phaseNs;
        m_summary.phaseMaxNs =
            m_summary.phaseCount ? std::max( m_summary.phaseMaxNs, m_summary.phaseNs ) : m_summary.phaseNs;
        ++m_summary.phaseCount;
    }

    if( counter == std::numeric_limits<uint64_t>::max() )
    {
        ++m_summary.invalid;
        m_summary.valid = false;
        m_planCount     = 0;
        next();
        finishEstimator( estimator );
        return;
    }

    if( m_intervalCount == m_config.window && m_pointCount == m_config.window )
    {
        if( !m_planCount )
        {
            const int64_t grid = predicted ? predicted : phase( counter );

            if( !commit( counter, grid, estimator.id ) )
            {
                next();
                finishEstimator( estimator );
                return;
            }
        }

        auto *current = plan( counter );

        if( current )
        {
            current->main   = observed;
            current->output = m_mainOutput ? m_outputTime : 0;

            if( !predicted )
            {
                m_summary.gridNs = current->grid;
                current->done[0] = true; // Startup never advertised this already-past passive preparation.
            }
            else
            {
                skip( *current, 0, &MagAOX::dmSyncoSimulation::Step::earlyMain );
            }
        }

        const uint64_t future = counter + 1;

        if( !plan( future ) )
        {
            const auto   &last          = m_plans[m_planCount - 1];
            int64_t       grid          = shifted( last.grid, last.counter, future, m_summary.periodNs );
            const int64_t candidate     = phase( future );
            estimator.nextCounter       = future;
            estimator.sourcePlanCounter = last.counter;
            estimator.sourceEstimatorId = last.estimatorId;
            estimator.sourceGridNs      = last.grid;
            estimator.baseGridNs        = grid;
            estimator.candidateNs       = candidate;

            if( grid && candidate )
            {
                const long double raw         = m_config.phaseGain * ( static_cast<long double>( candidate ) - grid );
                const int64_t     delta       = int64_t( std::clamp( raw,
                                                           -static_cast<long double>( m_config.phaseSlew ),
                                                           static_cast<long double>( m_config.phaseSlew ) ) );
                estimator.rawCorrectionNs     = int64_t( raw );
                estimator.appliedCorrectionNs = delta;
                grid                          = shifted( grid, 0, 1, delta );
            }

            if( !commit( future, grid, estimator.id ) )
            {
                next();
                finishEstimator( estimator );
                return;
            }

            estimator.committed       = true;
            estimator.committedGridNs = grid;
        }

        m_summary.valid = true;
    }

    through( observed, true );
    finishEstimator( estimator );
}

inline void Simulation::combined( uint64_t                                      counter,
                                  uint64_t                                      primaryCounter,
                                  int64_t                                       observed,
                                  bool                                          coherent,
                                  const MagAOX::dmSyncoSimulation::Observation &context )
{
    if( !m_summary.enabled || observed <= 0 || observed < m_summary.nowNs )
    {
        ++m_summary.invalid;
        return;
    }

    through( observed, false );
    const bool progression = !m_haveCombined || ( counter > m_combinedCounter && counter - m_combinedCounter == 1 );

    if( m_haveCombined && counter == m_combinedCounter )
    {
        through( observed, true );
        return;
    }

    if( coherent && m_haveMain && primaryCounter > m_summary.counter )
    {
        overlap( primaryCounter,
                 counter,
                 observed,
                 prediction( primaryCounter ),
                 MagAOX::dmSyncoSimulation::OverlapOrigin::combined,
                 context );
    }

    m_haveCombined    = true;
    m_combinedCounter = counter;

    if( !progression || !coherent )
    {
        badOutput();
    }
    else if( !m_haveMain || primaryCounter > m_summary.counter )
    {
        bool repeated = false;

        for( size_t i = 0; i < m_linkCount; ++i )
        {
            repeated = repeated || m_links[i].counter == primaryCounter;
        }

        if( repeated || m_linkCount == m_links.size() )
        {
            badOutput();
        }
        else
        {
            m_links[m_linkCount++] = { primaryCounter, observed, true };
        }
    }
    else if( primaryCounter != m_summary.counter || m_mainOutput )
    {
        badOutput();
    }
    else
    {
        ++m_summary.associations;
        m_mainOutput = true;
        m_outputTime = observed;

        if( auto *entry = plan( primaryCounter ) )
        {
            entry->output = observed;
        }
    }

    through( observed, true );
}

inline const MagAOX::dmSyncoSimulation::Summary &Simulation::summary() const
{
    return m_summary;
}

inline bool Simulation::popOverlap( MagAOX::dmSyncoSimulation::OverlapEvent &event )
{
    if( !m_traceCount )
    {
        return false;
    }

    event       = m_traceEvents[m_traceHead];
    m_traceHead = ( m_traceHead + 1 ) % m_traceEvents.size();
    --m_traceCount;
    ++m_traceDrained;
    return true;
}

inline MagAOX::dmSyncoSimulation::TraceStatus Simulation::traceStatus() const
{
    MagAOX::dmSyncoSimulation::TraceStatus status;
    status.enabled  = m_summary.enabled;
    status.detected = m_traceDetected;
    status.drained  = m_traceDrained;
    status.dropped  = m_traceDropped;
    status.queued   = m_traceCount;
    return status;
}

inline bool Simulation::popEstimator( MagAOX::dmSyncoSimulation::EstimatorSample &sample )
{
    if( !m_estimatorCount )
    {
        return false;
    }

    sample          = m_estimatorEvents[m_estimatorHead];
    m_estimatorHead = ( m_estimatorHead + 1 ) % m_estimatorEvents.size();
    --m_estimatorCount;
    ++m_estimatorDrained;
    return true;
}

inline MagAOX::dmSyncoSimulation::EstimatorStatus Simulation::estimatorStatus() const
{
    MagAOX::dmSyncoSimulation::EstimatorStatus status;
    status.enabled       = m_summary.enabled;
    status.observed      = m_estimatorObserved;
    status.triggers      = m_estimatorTriggers;
    status.selected      = m_estimatorSelected;
    status.drained       = m_estimatorDrained;
    status.dropped       = m_estimatorDropped;
    status.queued        = m_estimatorCount;
    status.history       = m_estimatorHistoryCount;
    status.postRemaining = m_estimatorPostRemaining;
    return status;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
