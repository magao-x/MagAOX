/** \file dmSyncopationTiming.hpp
 * \brief Same-host monotonic timing record shared by syncopation and its providers.
 */
#ifndef app_dmSyncopationTiming_hpp
#define app_dmSyncopationTiming_hpp

#include <array>
#include <cstdint>
#include <ctime>

namespace MagAOX
{
namespace app
{
namespace synco
{

constexpr int     maxSteps      = 16; ///< Fixed record capacity; not the requested cadence.
constexpr int64_t layoutVersion = 1;  ///< Integer record layout version.

/// Offsets in the one-dimensional signed-int64 ImageStreamIO record.
enum Field
{
    version = 0,
    epoch,
    generation,
    state,
    steps,
    period,
    reference,
    expires,
    cycle,
    cutoffs,
    targets = cutoffs + maxSteps,
    words   = targets + maxSteps
};

/// Schedule validity; acquisition records allow providers to prepare the first step 0.
enum ScheduleState
{
    invalid   = 0,
    acquiring = 1,
    armed     = 2
};

using Record = std::array<int64_t, words>; ///< Coherent record, copied under ImageStreamIO write/counter checks.

/// Read the scheduling clock, shared only by processes on this host.
inline int64_t monotonicNs();

/// Convert absolute monotonic nanoseconds to a timespec.
inline timespec toTimespec( int64_t ns /**< [in] absolute nanoseconds */ );

/// Validate record identity, dimensions and bounded schedule lifetime.
inline bool validRecord( const Record &record /**< [in] coherent timing record */,
                         int64_t       now /**< [in] current same-host monotonic nanoseconds */ );

/// Select the earliest still-usable snapshot cutoff, or -1 if none remains.
inline int nextStep( const Record &record /**< [in] coherent timing record */,
                     int64_t       now /**< [in] current same-host monotonic nanoseconds */ );

inline int64_t monotonicNs()
{
    timespec ts{};
    if( clock_gettime( CLOCK_MONOTONIC, &ts ) != 0 )
        return 0;
    return int64_t( ts.tv_sec ) * 1000000000 + ts.tv_nsec;
}

inline timespec toTimespec( int64_t ns )
{
    return { ns / 1000000000, ns % 1000000000 };
}

inline bool validRecord( const Record &record, int64_t now )
{
    if( record[version] != layoutVersion || record[epoch] <= 0 || record[generation] <= 0 ||
        ( record[state] != acquiring && record[state] != armed ) || record[steps] < 1 || record[steps] > maxSteps ||
        record[period] <= 0 || record[period] > 1000000000 || record[reference] <= 0 || record[reference] > now ||
        record[expires] <= now || record[expires] - record[reference] > 3 * record[period] )
        return false;
    for( int j = 0; j < record[steps]; ++j )
    {
        if( record[cutoffs + j] != 0 &&
            ( record[cutoffs + j] >= record[targets + j] || record[targets + j] > record[expires] ) )
            return false;
    }
    return true;
}

inline int nextStep( const Record &record, int64_t now )
{
    if( !validRecord( record, now ) )
        return -1;
    int selected = -1;
    for( int j = 0; j < record[steps]; ++j )
    {
        if( record[cutoffs + j] > now && ( selected < 0 || record[cutoffs + j] < record[cutoffs + selected] ) )
            selected = j;
    }
    return selected;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
