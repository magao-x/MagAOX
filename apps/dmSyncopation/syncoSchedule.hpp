/** \file syncoSchedule.hpp
 * \brief Deterministic scheduling and bounded wait feedback for dmSyncopation.
 */
#ifndef dmSyncopation_syncoSchedule_hpp
#define dmSyncopation_syncoSchedule_hpp

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "../../libMagAOX/app/dmSyncopationTiming.hpp"

namespace MagAOX
{
namespace app
{
namespace synco
{

/// Bounds in nanoseconds, independent of image timestamps and wall-clock time.
struct Bounds
{
    int64_t keepout{ 115000 };   ///< Controller processing through the last destination read.
    int64_t guard{ 20000 };      ///< Extra conservative reserve beyond qualified path/reader bounds.
    int64_t delivery{ 20000 };   ///< Maximum publication-to-controller delivery allowance.
    int64_t observation{ 5000 }; ///< Maximum publication-to-callback observation lag.
    int64_t early{ 20000 };      ///< Earliest-main allowance relative to the predicted observation.
    int64_t lead{ 20000 };       ///< Nominal passive-copy completion lead.
    int64_t visibility{ 1000 };  ///< Positive margin before earliest main publication.
    int64_t reserve{ 2000 };     ///< Preparation target precedes the hard copy cutoff.
    int64_t snapshot{ 10000 };   ///< One-pass whole-source snapshot budget.
    int64_t copy{ 10000 };       ///< Whole-destination copy/publication budget.
};

/// Distinct passive-copy outcomes; source silence is unrelated to this result.
enum class StepOutcome
{
    copiedOnTime,
    notStarted,
    copiedLate
};

/// Compute the hard passive-copy cutoff for a predicted next observed main event.
inline int64_t copyCutoff( int64_t       next /**< [in] predicted next callback time */,
                           const Bounds &bounds /**< [in] qualified timing bounds */ );

/// Test full reader, copy and wraparound feasibility for a fixed requested count.
inline bool feasible( int64_t       period /**< [in] planned period */,
                      int           count /**< [in] requested steps */,
                      const Bounds &bounds /**< [in] qualified timing bounds */ );

/// Report the largest feasible count without silently changing the requested count.
inline int maximumSteps( int64_t       period /**< [in] planned period */,
                         const Bounds &bounds /**< [in] qualified timing bounds */ );

/// Classify a passive copy independently of the quality-guideline residual.
inline StepOutcome classifyStep( bool    copied /**< [in] whether a copy started */,
                                 int64_t completed /**< [in] actual completion time */,
                                 int64_t cutoff /**< [in] effective hard completion cutoff */ );

/// Update a wait advance with clamping and no learning from invalid samples.
inline int64_t feedback( int64_t advance /**< [in] previous wait advance */,
                         int64_t residual /**< [in] actual minus target */,
                         double  gain /**< [in] integrator gain */,
                         int64_t limit /**< [in] maximum absolute advance */,
                         bool    valid /**< [in] whether this event is a usable sample */ );

inline int64_t copyCutoff( int64_t next, const Bounds &b )
{
    return next - std::max( b.lead, b.early + b.observation + b.visibility );
}

inline bool feasible( int64_t period, int count, const Bounds &b )
{
    if( period <= 0 || period > 1000000000 || count < 1 || count > maxSteps || b.keepout < 0 || b.guard < 0 ||
        b.delivery < 0 || b.observation < 0 || b.early < 0 || b.lead < 0 || b.visibility <= 0 || b.reserve <= 0 ||
        b.snapshot <= 0 || b.copy <= 0 )
        return false;
    const int64_t gap  = period / count;
    const int64_t hold = b.delivery + b.keepout + b.guard;
    const int64_t tail = std::max( b.lead, b.early + b.observation + b.visibility );
    // A main is observed after publication; waiting from observation is conservative.
    return gap >= hold + b.copy && gap >= hold + b.copy + tail + b.reserve && gap > b.snapshot + b.copy + b.reserve;
}

inline int maximumSteps( int64_t period, const Bounds &b )
{
    int result = 0;
    for( int count = 1; count <= maxSteps; ++count )
        if( feasible( period, count, b ) )
            result = count;
    return result;
}

inline StepOutcome classifyStep( bool copied, int64_t completed, int64_t cutoff )
{
    return !copied ? StepOutcome::notStarted
                   : ( completed <= cutoff ? StepOutcome::copiedOnTime : StepOutcome::copiedLate );
}

inline int64_t feedback( int64_t advance, int64_t residual, double gain, int64_t limit, bool valid )
{
    if( !valid )
        return advance;
    const double corrected = double( advance ) + gain * double( residual );
    return int64_t( std::clamp( corrected, -double( limit ), double( limit ) ) );
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
