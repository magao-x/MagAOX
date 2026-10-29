/** \file syncoSimulation.hpp
 * \brief Bounded, event-driven prediction of syncopation policy without stream access or DM writes.
 */
#ifndef dmSyncopation_syncoSimulation_hpp
#define dmSyncopation_syncoSimulation_hpp

#include <array>
#include <limits>

#include <mx/math/floatUtils.hpp>

#include "syncoSchedule.hpp"
#include "syncoCadence.hpp"
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
        unsigned window{ 32 };            ///< Retained diagnostic intervals and robust phase observations.
        int64_t  nominalPeriod{ 500000 }; ///< Initial period in monotonic nanoseconds.
        int64_t  minInterval{ 250000 };   ///< Smallest eligible diagnostic interval or long-span cycle period.
        int64_t  maxInterval{ 2000000 };  ///< Largest eligible diagnostic interval or long-span cycle period.
        double   phaseGain{ 0.1 };        ///< Bounded correction toward the robust phase candidate.
        int64_t  phaseSlew{ 1000 };       ///< Maximum correction per newly committed cycle.
        int64_t  cadenceMinPeriod{ 0 };   ///< Minimum whole-cycle period admitted by the nominal cadence rating.
        int64_t  timeout{ 4000000 };      ///< Maximum primary silence before clearing cadence and phase anchors.
        int64_t  cadenceWindowNs{ 10000000000LL }; ///< Requested monotonic counter/time baseline.
        int64_t  cadenceUpdateNs{ 1000000000LL };  ///< Minimum interval between cadence refreshes.
        bool     readerReleaseReady{ false }; ///< Use exact observed main reader release instead of a fixed main hold.
        int64_t  virtualReaderNs{ 30000 };    ///< Assumed auxiliary trigger-to-last-reader duration.
        int64_t  backendNs{ 155000 };         ///< Assumed output/trigger-to-whole-backend-ready duration.
    };

    /// Validate configuration and begin a fresh simulation with empty estimators.
    bool configure( const Config &config /**< [in] fixed startup model */ );

    /// Begin a new session with empty model history, commitments and conditional counters.
    void reset();

    /// Observe one coherent main; positive clock regression starts a new session, unlike rejected late
    /// output/evaluation calls.
    void
    primary( uint64_t counter /**< [in] genuine primary counter */,
             int64_t  observed /**< [in] coherent same-host monotonic observation */,
             int64_t  publicationInterval = 0 /**< [in] producer-clock diagnostic interval, ns; zero unavailable */,
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

    /// Retain exact early-release evidence without advancing, retiming or revisiting any committed occurrence.
    void release( const MagAOX::dmSyncoSimulation::Readiness &event /**< [in] Coherent observer capture. */ );

    /// Revoke all retained release evidence after a lifecycle or transport-generation change.
    void resetRelease();

    /// Revoke evidence and retain the exact number of transport handoffs lost.
    void releaseDrop( uint64_t count = 1 /**< [in] Newly dropped transport handoffs. */ );

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
    /// Frozen combined input identity whose release must match exactly.
    struct Command
    {
        uint64_t inode{ 0 };         ///< Combined stream inode.
        uint64_t counter{ 0 };       ///< Combined publication counter.
        int64_t  publicationNs{ 0 }; ///< Raw producer publication time; used only for exact identity.
    };

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
        int64_t                       period{ 0 };      ///< Conservative whole-nanosecond period for feasibility.
        uint64_t                      periodQ32{ 0 };   ///< Exact fractional period frozen with this cycle.
        int64_t                       fractionQ32{ 0 }; ///< Signed fractional remainder of the frozen main prediction.
        int64_t                       main{ 0 };        ///< Actual associated primary observation, or zero.
        int64_t                       output{ 0 };      ///< Coherent combined observation, or zero.
        Command                       outputKey;        ///< Exact combined identity associated with this plan.
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
        Command  key;            ///< Exact combined identity retained before its primary is observed.
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
    int64_t m_virtualBackend{ 0 }; ///< Assumed backend-ready time for the latest hypothetical auxiliary command.
    MagAOX::dmSyncoSimulation::OverlapEvent
            m_backendPlan;   ///< Last active backend provenance, retained through passive copies.
    Command m_mainOutputKey; ///< Exact current actual-main combined identity.
    std::array<MagAOX::dmSyncoSimulation::Readiness, 16> m_releases{};       ///< Fixed exact-release history.
    size_t                                               m_releaseNext{ 0 }; ///< Next release-history replacement.
    uint64_t                     m_releaseSourceEpoch{ 0 }; ///< Highest accepted app-owned source epoch.
    MagAOX::dmCompletion::Record m_releaseRecord; ///< Last observed release attempt for progression/enrichment checks.
    int64_t m_releaseObserved{ 0 }; ///< Last distinct release observation for monotonic source progression.
    bool m_releaseFault{ false };   ///< Explicit SDK failure closes readiness until a source or producer epoch changes.

    /// Refresh nominal full-cycle diagnostics without suppressing individually feasible readiness-mode slots.
    void feasibility( int64_t period /**< [in] Conservative period in whole nanoseconds. */ );

    /// Return the exact retained reader-release capture, or null when unavailable.
    const MagAOX::dmSyncoSimulation::Readiness *
    released( const Command &key /**< [in] Frozen combined identity. */ ) const;

    /// Clear only retained release identities while retaining source-progression fences.
    void clearRelease();
    std::array<Point, MagAOX::dmSyncoSimulation::cadenceAnchorCapacity>
             m_cadenceAnchors{};     ///< Slow counter/time anchors.
    size_t   m_cadenceCount{ 0 };    ///< Retained cadence anchors.
    size_t   m_cadenceNext{ 0 };     ///< Next anchor replacement.
    uint64_t m_cadenceEpoch{ 0 };    ///< Process-lifetime reset/silence epoch identity.
    int64_t  m_copyBegin{ 0 };       ///< Last modeled copy's start, retained for retrospective overlap detection.
    int64_t  m_copyEnd{ 0 };         ///< Last modeled copy's completion/trigger time.
    int      m_copyStep{ 0 };        ///< Step associated with the last modeled copy/reader hold.
    uint64_t m_copyCounter{ 0 };     ///< Genuine counter labelling the last modeled copy's cycle.
    int64_t  m_copyGrid{ 0 };        ///< Frozen prediction for the last modeled copy's cycle.
    int64_t  m_copyPeriod{ 0 };      ///< Rounded diagnostic period frozen with the last modeled copy's cycle.
    uint64_t m_copyPeriodQ32{ 0 };   ///< Exact period associated with the last modeled copy.
    int64_t  m_copyFractionQ32{ 0 }; ///< Frozen main target fractional remainder for the last copy.
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

    /// Clear slow counter/time anchors and restore the nominal fractional period.
    void clearCadence();

    /// Incorporate one distinct coherent primary; endpoint refreshes occur only at the configured slow interval.
    void updateCadence( uint64_t counter /**< [in] genuine primary counter */,
                        int64_t  observed /**< [in] monotonic accepted-primary observation */ );

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
                 uint64_t estimatorId /**< [in] accepted-primary update committing this plan */,
                 int64_t  fractionQ32 = 0 /**< [in] signed fractional main-target remainder */ );

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

    /// Count a main observation that could overlap the last modeled copy, active reader or backend interval.
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

inline void Simulation::feasibility( int64_t period )
{
    if( !m_config.readerReleaseReady )
    {
        m_summary.feasible = feasible( period, m_config.steps, m_config.bounds ) && period >= m_config.cadenceMinPeriod;
        return;
    }

    const auto   &b    = m_config.bounds;
    const int64_t gap  = period / m_config.steps;
    const int64_t tail = std::max( b.lead, b.early + b.observation + b.visibility );
    m_summary.readerCycleFeasible =
        gap >= m_config.virtualReaderNs + b.guard + b.copy + tail + b.reserve && gap > b.snapshot + b.copy + b.reserve;
    m_summary.backendCycleFeasible =
        gap >= m_config.backendNs && ( m_config.steps == 1 || gap >= m_config.backendNs + b.early + b.observation );
    m_summary.feasible =
        m_summary.readerCycleFeasible && m_summary.backendCycleFeasible && period >= m_config.cadenceMinPeriod;
}

inline void Simulation::clearRelease()
{
    m_releases              = {};
    m_releaseNext           = 0;
    m_summary.latestRelease = {};
}

inline void Simulation::resetRelease()
{
    clearRelease();
    ++m_summary.readiness.sourceResets;
}

inline void Simulation::releaseDrop( uint64_t count )
{
    if( count )
    {
        resetRelease();
        m_summary.readiness.queueDrops += count;
    }
}

inline const MagAOX::dmSyncoSimulation::Readiness *Simulation::released( const Command &key ) const
{
    if( !key.inode || !key.counter || key.publicationNs <= 0 || m_releaseFault )
    {
        return nullptr;
    }

    for( const auto &event : m_releases )
    {
        const auto    &record = event.record;
        const __int128 stamp  = __int128( record.publicationSec ) * 1000000000 + record.publicationNsec;

        if( event.observedNs > 0 && record.inputInode == key.inode && record.inputCounter == key.counter &&
            stamp == key.publicationNs )
        {
            return &event;
        }
    }

    return nullptr;
}

inline void Simulation::release( const MagAOX::dmSyncoSimulation::Readiness &event )
{
    if( !m_summary.enabled || !m_config.readerReleaseReady )
    {
        return;
    }

    auto &quality = m_summary.readiness;
    ++quality.observed;
    const auto &record = event.record;

    if( event.sourceEpoch && event.sourceEpoch >= m_releaseSourceEpoch && event.observedNs > 0 &&
        event.observedNs >= m_summary.nowNs )
    {
        through( event.observedNs, false ); // Apply revocation only after earlier fixed deadlines were evaluated.
    }

    if( !event.sourceEpoch || event.sourceEpoch < m_releaseSourceEpoch )
    {
        ++quality.invalid;
        return;
    }

    if( event.sourceEpoch > m_releaseSourceEpoch )
    {
        resetRelease();
        m_releaseSourceEpoch = event.sourceEpoch;
        m_releaseRecord      = {};
        m_releaseObserved    = 0;
        m_releaseFault       = false;
    }

    if( !event.coherent )
    {
        clearRelease();
        ++quality.incoherent;
        return;
    }

    const __int128 stamp = __int128( record.publicationSec ) * 1000000000 + record.publicationNsec;
    const bool     base  = record.kind == MagAOX::dmCompletion::Kind::readers && record.session && record.epoch &&
                      record.sequence && record.inputInode && record.inputCounter && record.publisherPid > 0 &&
                      record.publicationSec > 0 && record.publicationNsec >= 0 && record.publicationNsec < 1000000000 &&
                      stamp <= std::numeric_limits<int64_t>::max() && record.startedNs > 0 &&
                      record.finishedNs >= record.startedNs && event.observedNs >= record.finishedNs &&
                      event.observedNs > 0 && MagAOX::dmCompletion::readersQualified( record );

    if( !base || ( m_releaseRecord.session && ( record.session < m_releaseRecord.session ||
                                                ( record.session == m_releaseRecord.session &&
                                                  ( record.publisherPid != m_releaseRecord.publisherPid ||
                                                    record.epoch < m_releaseRecord.epoch ) ) ) ) )
    {
        clearRelease();
        ++quality.invalid;
        return;
    }

    if( m_releaseRecord.session &&
        ( record.session > m_releaseRecord.session || record.epoch > m_releaseRecord.epoch ) )
    {
        resetRelease();
        m_releaseRecord   = {};
        m_releaseObserved = 0;
        m_releaseFault    = false;
    }

    if( record.outcome != MagAOX::dmTiming::Outcome::unknown && record.outcome != MagAOX::dmTiming::Outcome::success )
    {
        clearRelease();
        ++quality.sdkFailed;
        m_releaseFault  = true;
        m_releaseRecord = record;
        return;
    }

    if( ( record.outcome == MagAOX::dmTiming::Outcome::unknown &&
          record.backendResult != MagAOX::dmCompletion::unavailableResult ) ||
        ( record.outcome == MagAOX::dmTiming::Outcome::success && record.backendResult != 0 &&
          record.backendResult != MagAOX::dmCompletion::unavailableResult ) ||
        m_releaseFault )
    {
        clearRelease();
        ++quality.invalid;
        return;
    }

    if( m_releaseRecord.sequence && record.sequence <= m_releaseRecord.sequence )
    {
        const auto &previous = m_releaseRecord;
        const bool  same     = record.sequence == previous.sequence && record.inputInode == previous.inputInode &&
                          record.inputCounter == previous.inputCounter &&
                          record.publicationSec == previous.publicationSec &&
                          record.publicationNsec == previous.publicationNsec &&
                          record.startedNs == previous.startedNs && record.finishedNs == previous.finishedNs;

        if( !same )
        {
            clearRelease();
            ++quality.identityMismatch;
        }
        else if( previous.outcome == MagAOX::dmTiming::Outcome::unknown &&
                 record.outcome == MagAOX::dmTiming::Outcome::success )
        {
            m_releaseRecord = record;
            return;
        }

        ++quality.duplicates;
        return;
    }

    const bool gap = m_releaseRecord.sequence && ( record.sequence != m_releaseRecord.sequence + 1 ||
                                                   record.inputCounter > m_releaseRecord.inputCounter + 1 );
    const bool regression =
        m_releaseRecord.sequence &&
        ( event.observedNs <= m_releaseObserved || record.inputInode != m_releaseRecord.inputInode ||
          record.inputCounter <= m_releaseRecord.inputCounter || record.startedNs <= m_releaseRecord.startedNs ||
          record.finishedNs <= m_releaseRecord.finishedNs );
    m_releaseRecord   = record;
    m_releaseObserved = event.observedNs;

    if( gap || regression )
    {
        clearRelease();
        quality.gaps += gap;
        quality.invalid += regression;
        return;
    }

    m_releases[m_releaseNext] = event;
    m_releaseNext             = ( m_releaseNext + 1 ) % m_releases.size();
    m_summary.latestRelease   = event;
    ++quality.qualified;
}

inline void Simulation::clearCadence()
{
    m_cadenceCount = m_cadenceNext = 0;
    m_summary.cadence              = {};
    m_summary.cadence.periodQ32    = uint64_t( m_config.nominalPeriod ) * MagAOX::dmSyncoSimulation::periodScale;
    m_summary.cadence.epoch        = ++m_cadenceEpoch;
    m_summary.cadence.windowNs     = m_config.cadenceWindowNs;
    m_summary.cadence.updateNs     = m_config.cadenceUpdateNs;
    m_summary.periodNs             = m_config.nominalPeriod;
    feasibility( m_config.nominalPeriod );
}

inline void Simulation::updateCadence( uint64_t counter, int64_t observed )
{
    auto &cadence         = m_summary.cadence;
    cadence.observationNs = observed;

    if( m_cadenceCount )
    {
        const size_t latest = ( m_cadenceNext + m_cadenceAnchors.size() - 1 ) % m_cadenceAnchors.size();

        if( observed - m_cadenceAnchors[latest].observed < m_config.cadenceUpdateNs )
        {
            return;
        }

        const size_t  oldest = ( m_cadenceNext + m_cadenceAnchors.size() - m_cadenceCount ) % m_cadenceAnchors.size();
        const Point  *anchor = &m_cadenceAnchors[oldest];
        const int64_t cutoff = observed - m_config.cadenceWindowNs;

        for( size_t offset = 1; offset < m_cadenceCount; ++offset )
        {
            const auto &candidate = m_cadenceAnchors[( oldest + offset ) % m_cadenceAnchors.size()];

            if( candidate.observed <= cutoff )
            {
                anchor = &candidate;
            }
            else
            {
                break;
            }
        }

        const int64_t  elapsed = observed - anchor->observed;
        const uint64_t cycles  = counter > anchor->counter ? counter - anchor->counter : 0;
        const __int128 scaled =
            cycles ? ( __int128( elapsed ) * MagAOX::dmSyncoSimulation::periodScale + cycles / 2 ) / cycles : 0;
        const __int128 minimum = __int128( m_config.minInterval ) * MagAOX::dmSyncoSimulation::periodScale;
        const __int128 maximum = __int128( m_config.maxInterval ) * MagAOX::dmSyncoSimulation::periodScale;

        if( elapsed >= m_config.cadenceUpdateNs && cycles && scaled >= minimum && scaled <= maximum )
        {
            cadence.periodQ32 = uint64_t( scaled );
            cadence.spanNs    = elapsed;
            cadence.cycles    = cycles;
            cadence.updatedNs = observed;
            cadence.state     = elapsed >= m_config.cadenceWindowNs ? MagAOX::dmSyncoSimulation::CadenceState::full
                                                                    : MagAOX::dmSyncoSimulation::CadenceState::growing;
            ++cadence.updates;
            m_summary.periodNs = int64_t( ( cadence.periodQ32 + MagAOX::dmSyncoSimulation::periodScale / 2 ) /
                                          MagAOX::dmSyncoSimulation::periodScale );
        }
        else
        {
            ++cadence.rejected;
        }
    }

    m_cadenceAnchors[m_cadenceNext] = { counter, observed };
    m_cadenceNext                   = ( m_cadenceNext + 1 ) % m_cadenceAnchors.size();
    m_cadenceCount                  = std::min( m_cadenceCount + 1, m_cadenceAnchors.size() );
    cadence.anchors                 = m_cadenceCount;
}

inline bool Simulation::configure( const Config &config )
{
    if( config.steps < 1 || config.steps > maxSteps || config.window < 2 || config.window > m_intervals.size() ||
        config.minInterval <= 0 || config.maxInterval < config.minInterval || config.maxInterval > 1000000000 ||
        config.nominalPeriod < config.minInterval || config.nominalPeriod > config.maxInterval ||
        !mx::math::isFinite( config.phaseGain ) || config.phaseGain < 0 || config.phaseGain > 1 ||
        config.phaseSlew < 0 || config.phaseSlew > config.minInterval / 4 ||
        config.timeout < 2 * config.nominalPeriod || config.timeout > 1000000000000LL || config.cadenceMinPeriod < 0 ||
        config.cadenceMinPeriod > 1000000000000LL || config.cadenceUpdateNs <= 0 ||
        config.cadenceWindowNs < config.cadenceUpdateNs || config.cadenceWindowNs > 1000000000000LL ||
        config.virtualReaderNs <= 0 || config.virtualReaderNs > 1000000000 ||
        config.backendNs < config.virtualReaderNs || config.backendNs > 1000000000 ||
        __int128( config.cadenceWindowNs ) >
            __int128( MagAOX::dmSyncoSimulation::cadenceAnchorCapacity - 1 ) * config.cadenceUpdateNs )
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
    const uint64_t session       = m_summary.session + 1;
    m_summary                    = {};
    m_summary.session            = session;
    m_summary.resets             = 1;
    m_summary.enabled            = true;
    m_summary.valid              = false;
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
    m_summary.readerReleaseReady = m_config.readerReleaseReady;
    m_summary.virtualReaderNs    = m_config.virtualReaderNs;
    m_summary.backendNs          = m_config.backendNs;
    m_intervalCount = m_intervalNext = m_pointCount = m_pointNext = m_planCount = 0;
    m_linkCount                                                                 = 0;
    m_haveMain = m_haveCombined = m_mainOutput = false;
    m_outputTime = m_virtualHold = m_copyBegin = m_copyEnd = 0;
    m_virtualBackend                                       = 0;
    m_backendPlan                                          = {};
    m_mainOutputKey                                        = {};
    clearRelease();
    m_releaseSourceEpoch = 0;
    m_releaseRecord      = {};
    m_releaseObserved    = 0;
    m_releaseFault       = false;
    clearCadence();
    m_copyStep    = 0;
    m_copyCounter = 0;
    m_copyGrid = m_copyPeriod = m_copyFractionQ32 = 0;
    m_copyPeriodQ32                               = 0;
    m_copyEstimatorId                             = 0;
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
        return shiftGrid( { last.grid, last.fractionQ32 }, last.counter, counter, last.periodQ32 ).ns;
    }

    return 0;
}

inline int64_t Simulation::phase( uint64_t counter )
{
    for( size_t i = 0; i < m_pointCount; ++i )
    {
        m_scratch[i] =
            shiftGrid( { m_points[i].observed, 0 }, m_points[i].counter, counter, m_summary.cadence.periodQ32 ).ns;

        if( !m_scratch[i] )
        {
            return 0;
        }
    }

    return m_pointCount ? median( m_pointCount ) : 0;
}

inline bool Simulation::commit( uint64_t counter, int64_t grid, uint64_t estimatorId, int64_t fractionQ32 )
{
    if( m_planCount == m_plans.size() || !grid ||
        !shiftGrid( { grid, fractionQ32 }, 0, 2, m_summary.cadence.periodQ32 ).ns )
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
    entry.periodQ32   = m_summary.cadence.periodQ32;
    entry.period      = int64_t( entry.periodQ32 / MagAOX::dmSyncoSimulation::periodScale );
    entry.fractionQ32 = fractionQ32;
    entry.targets[0]  = copyCutoff( grid, m_config.bounds ) - m_config.bounds.reserve;
    entry.starts[0]   = entry.targets[0] - m_config.bounds.copy;

    for( int j = 1; j < m_config.steps; ++j )
    {
        entry.targets[j] = stepGrid( { grid, fractionQ32 }, entry.periodQ32, j, m_config.steps ).ns;
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
    sample.cadence                             = m_summary.cadence;
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

    if( ( !m_config.readerReleaseReady && !feasible( entry.period, m_config.steps, b ) ) ||
        entry.period < m_config.cadenceMinPeriod || start <= 0 )
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
    const __int128 hold  = m_config.readerReleaseReady ? 0 : __int128( actual ) + b.delivery + b.keepout + b.guard;

    if( m_config.readerReleaseReady )
    {
        const auto *evidence = released( step ? entry.outputKey : m_mainOutputKey );

        if( !evidence )
        {
            skip( entry, step, &Outcome::releaseMissing );
            return;
        }

        if( __int128( evidence->observedNs ) + b.guard > start )
        {
            skip( entry, step, &Outcome::releaseLate );
            return;
        }

        if( step && ( __int128( actual ) + m_config.backendNs > target || m_virtualBackend > target ) )
        {
            skip( entry, step, &Outcome::backendBusy );
            return;
        }
    }

    if( hold > start || m_virtualHold > start )
    {
        skip( entry, step, &Outcome::readerHold );
        return;
    }

    const auto   *following = plan( entry.counter + 1 );
    const int64_t nextGrid =
        following ? following->grid : shiftGrid( { entry.grid, entry.fractionQ32 }, 0, 1, entry.periodQ32 ).ns;
    const __int128 readerEnd = __int128( target ) + ( m_config.readerReleaseReady ? m_config.virtualReaderNs + b.guard
                                                                                  : b.delivery + b.keepout + b.guard );

    if( m_config.readerReleaseReady && step &&
        ( !nextGrid || __int128( target ) + m_config.backendNs > __int128( nextGrid ) - b.early - b.observation ) )
    {
        skip( entry, step, &Outcome::backendWrap );
        return;
    }

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
    m_copyPeriod      = int64_t( ( entry.periodQ32 + MagAOX::dmSyncoSimulation::periodScale / 2 ) /
                            MagAOX::dmSyncoSimulation::periodScale );
    m_copyPeriodQ32   = entry.periodQ32;
    m_copyFractionQ32 = entry.fractionQ32;
    m_copyEstimatorId = entry.estimatorId;

    if( step )
    {
        m_virtualHold    = int64_t( readerEnd );
        m_virtualBackend = m_config.readerReleaseReady ? target + m_config.backendNs : 0;

        if( m_config.readerReleaseReady )
        {
            m_backendPlan                 = {};
            m_backendPlan.step            = step;
            m_backendPlan.planCounter     = m_copyCounter;
            m_backendPlan.planEstimatorId = m_copyEstimatorId;
            m_backendPlan.planGridNs      = m_copyGrid;
            m_backendPlan.periodNs        = m_copyPeriod;
            m_backendPlan.periodQ32       = m_copyPeriodQ32;
            m_backendPlan.planFractionQ32 = m_copyFractionQ32;
            m_backendPlan.copyBeginNs     = m_copyBegin;
            m_backendPlan.copyEndNs       = m_copyEnd;
            m_backendPlan.readerEndNs     = m_virtualHold;
            m_backendPlan.backendEndNs    = m_virtualBackend;
        }
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
    const bool copying = m_copyBegin && observed >= m_copyBegin && observed < m_copyEnd;
    const bool reading = m_copyBegin && m_copyStep > 0 && observed >= m_copyEnd && observed < m_virtualHold;
    const bool backend = m_config.readerReleaseReady && m_backendPlan.backendEndNs > 0 &&
                         observed >= m_backendPlan.readerEndNs && observed < m_backendPlan.backendEndNs;
    const bool backendOnly = !copying && !reading && backend;

    if( copying || reading || backend )
    {
        ++m_summary.outcomes[backendOnly ? m_backendPlan.step : m_copyStep].overlap;
        ++m_traceDetected;
        triggerEstimator();

        if( m_traceCount == m_traceEvents.size() )
        {
            ++m_traceDropped;
            return;
        }

        auto &event = m_traceEvents[( m_traceHead + m_traceCount ) % m_traceEvents.size()];
        event       = backendOnly ? m_backendPlan : MagAOX::dmSyncoSimulation::OverlapEvent{};

        if( !backendOnly )
        {
            event.planCounter     = m_copyCounter;
            event.planEstimatorId = m_copyEstimatorId;
            event.step            = m_copyStep;
            event.planGridNs      = m_copyGrid;
            event.periodNs        = m_copyPeriod;
            event.periodQ32       = m_copyPeriodQ32;
            event.planFractionQ32 = m_copyFractionQ32;
            event.copyBeginNs     = m_copyBegin;
            event.copyEndNs       = m_copyEnd;
            event.readerEndNs     = m_copyStep > 0 ? m_virtualHold : 0;
            event.backendEndNs    = m_config.readerReleaseReady && m_copyStep > 0 ? m_virtualBackend : 0;
        }

        event.id              = m_traceDetected;
        event.session         = m_summary.session;
        event.mainCounter     = mainCounter;
        event.combinedCounter = combinedCounter;
        event.lastMainCounter = m_summary.counter;
        event.lastEstimatorId = m_estimatorObserved;
        event.origin          = origin;
        event.window          = copying ? MagAOX::dmSyncoSimulation::OverlapWindow::copy
                                        : ( reading ? MagAOX::dmSyncoSimulation::OverlapWindow::readerHold
                                                    : MagAOX::dmSyncoSimulation::OverlapWindow::backend );
        event.observedNs      = observed;
        event.mainGridNs      = mainGrid;
        event.lastMainNs      = m_summary.lastMainNs;
        event.lastOutputNs    = m_outputTime;
        event.observation     = context;
        ++m_traceCount;
    }
}

inline void Simulation::badOutput()
{
    ++m_summary.outputGaps;
    m_mainOutput    = false;
    m_mainOutputKey = {};

    if( auto *entry = plan( m_summary.counter ) )
    {
        entry->output    = 0;
        entry->outputKey = {};
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

    const bool staleMain = m_haveMain && now - m_summary.lastMainNs > m_config.timeout;

    if( staleMain )
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
        clearCadence();
        m_haveMain              = false;
        m_summary.periodSamples = 0;
        m_linkCount             = 0;
        m_haveCombined = m_mainOutput = false;
        m_outputTime                  = 0;
        m_virtualHold = m_copyBegin = m_copyEnd = 0;
        m_virtualBackend                        = 0;
        m_backendPlan                           = {};
        m_mainOutputKey                         = {};
        if( m_config.readerReleaseReady )
        {
            resetRelease();
        }
        m_pointCount = m_pointNext = 0;
        m_summary.phaseSamples     = 0;
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
    if( !m_summary.enabled || observed <= 0 )
    {
        ++m_summary.invalid;
        return;
    }

    if( observed < m_summary.nowNs )
    {
        reset();
        ++m_summary.invalid;
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
    estimator.periodBeforeQ32       = m_summary.cadence.periodQ32;
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
            m_mainOutput    = true;
            m_outputTime    = m_links[i].observed;
            m_mainOutputKey = m_links[i].key;
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

    updateCadence( counter, observed );
    const int64_t conservativePeriod = int64_t( m_summary.cadence.periodQ32 / MagAOX::dmSyncoSimulation::periodScale );
    feasibility( conservativePeriod );

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

    if( m_pointCount == m_config.window )
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
            current->main      = observed;
            current->output    = m_mainOutput ? m_outputTime : 0;
            current->outputKey = m_mainOutputKey;

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
            const auto &last = m_plans[m_planCount - 1];
            const auto  shifted =
                shiftGrid( { last.grid, last.fractionQ32 }, last.counter, future, m_summary.cadence.periodQ32 );
            int64_t       grid          = shifted.ns;
            const int64_t candidate     = phase( future );
            estimator.nextCounter       = future;
            estimator.sourcePlanCounter = last.counter;
            estimator.sourceEstimatorId = last.estimatorId;
            estimator.sourceGridNs      = last.grid;
            estimator.sourceFractionQ32 = last.fractionQ32;
            estimator.baseFractionQ32   = shifted.fractionQ32;
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
                grid = delta >= 0 ? ( grid <= std::numeric_limits<int64_t>::max() - delta ? grid + delta : 0 )
                                  : ( grid > -delta ? grid + delta : 0 );
            }

            if( !commit( future, grid, estimator.id, shifted.fractionQ32 ) )
            {
                next();
                finishEstimator( estimator );
                return;
            }

            estimator.committed            = true;
            estimator.committedGridNs      = grid;
            estimator.committedFractionQ32 = shifted.fractionQ32;
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
            m_links[m_linkCount++] = {
                primaryCounter, observed, true, { context.streamInode, counter, context.publicationNs } };
        }
    }
    else if( primaryCounter != m_summary.counter || m_mainOutput )
    {
        badOutput();
    }
    else
    {
        ++m_summary.associations;
        m_mainOutput    = true;
        m_outputTime    = observed;
        m_mainOutputKey = { context.streamInode, counter, context.publicationNs };

        if( auto *entry = plan( primaryCounter ) )
        {
            entry->output    = observed;
            entry->outputKey = m_mainOutputKey;
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
