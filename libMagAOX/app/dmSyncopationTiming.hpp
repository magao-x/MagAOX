/** \file dmSyncopationTiming.hpp
 * \brief Same-host monotonic timing record shared by syncopation and its providers.
 */
#ifndef app_dmSyncopationTiming_hpp
#define app_dmSyncopationTiming_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <ctime>

namespace MagAOX
{
namespace app
{
namespace synco
{

constexpr int     maxSteps          = 16; ///< Fixed record capacity; not the requested cadence.
constexpr int64_t layoutVersion     = 1;  ///< Legacy persistent-image schedule version.
constexpr int64_t productionVersion = 2;  ///< Tagged independent-producer schedule version; same schedule layout.

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

/// One immutable advertised command slot, identified independently of notification count.
struct Request
{
    int64_t epoch{ 0 };      ///< Scheduler restart identity.
    int64_t generation{ 0 }; ///< Scheduler plan generation.
    int64_t cycle{ 0 };      ///< Genuine main label associated with this target.
    int     step{ -1 };      ///< Passive step zero or positive auxiliary subdivision.
    int64_t targetNs{ 0 };   ///< Frozen same-host monotonic counter-publication target.
    int64_t cutoffNs{ 0 };   ///< Latest complete source-payload publication.
};

/// Producer-owned commit sidecar fields; values describe one immutable source image.
enum CommandField
{
    cmdVersion = 0,
    cmdEpoch,
    cmdGeneration,
    cmdCycle,
    cmdStep,
    cmdTarget,
    cmdCutoff,
    cmdPayloadCounter,
    cmdReadyNs,
    cmdSession,
    cmdSequence,
    cmdPublisher,
    cmdPayloadInode,
    commitWords
};

using CommitRecord = std::array<int64_t, commitWords>; ///< Producer's coherent _ready record.

constexpr int ackStatus = commitWords;     ///< Exact commit echo followed by the transport disposition.
constexpr int ackWords  = commitWords + 1; ///< Number of signed-int64 acknowledgement words.

/// Acknowledgements release source ownership after snapshot/discard, never after physical DM consumption.
enum AckStatus
{
    copied    = 1, ///< Consumer copied the payload into private storage; not a DM delivery receipt.
    discarded = 2  ///< Consumer rejected or expired this command without using its payload.
};

using AckRecord = std::array<int64_t, ackWords>; ///< Consumer-owned _ack record, with one exact commit echo.

/// Read the scheduling clock, shared only by processes on this host.
inline int64_t monotonicNs();

/// Convert absolute monotonic nanoseconds to a timespec.
inline timespec toTimespec( int64_t ns /**< [in] absolute nanoseconds */ );

/// Validate record identity, dimensions and bounded schedule lifetime.
inline bool validRecord( const Record &record /**< [in] coherent timing record */,
                         int64_t       now /**< [in] current same-host monotonic nanoseconds */ );

/// Validate a schedule of the selected explicit version.
inline bool validRecordVersion( const Record &record /**< [in] Coherent timing record. */,
                                int64_t       now /**< [in] Current same-host monotonic nanoseconds. */,
                                int64_t       supportedVersion /**< [in] Required layout version. */ );

/// Validate a tagged-production schedule without accepting the legacy protocol.
inline bool validProductionRecord( const Record &record /**< [in] Coherent timing record. */,
                                   int64_t       now /**< [in] Current same-host monotonic nanoseconds. */ );

/// Select the earliest usable future production slot, or -1.
inline int nextProductionStep( const Record &record /**< [in] Coherent production timing record. */,
                               int64_t       now /**< [in] Current same-host monotonic nanoseconds. */ );

/// Extract a target's genuine cycle label; step zero belongs to the next main.
inline Request request( const Record &record /**< [in] Valid coherent schedule. */,
                        int           step /**< [in] Selected available step index. */ );

/// Check all producer identity and completion fields without trusting a stream notification.
inline bool validCommand( const CommitRecord &command /**< [in] Coherent command commit. */,
                          int64_t             now /**< [in] Current same-host monotonic nanoseconds. */ );

/// Match a committed payload to one exact scheduler slot.
inline bool commandMatches( const CommitRecord &command /**< [in] Coherent producer commit. */,
                            const Request      &slot /**< [in] Expected immutable target. */ );

/// Check that a consumer acknowledged this exact source publication.
inline bool acknowledges( const AckRecord    &ack /**< [in] Coherent consumer disposition. */,
                          const CommitRecord &command /**< [in] Outstanding producer commit. */ );

/// Select the earliest still-usable snapshot cutoff, or -1 if none remains.
inline int nextStep( const Record &record /**< [in] coherent timing record */,
                     int64_t       now /**< [in] current same-host monotonic nanoseconds */ );

inline int64_t monotonicNs()
{
    timespec ts{};

    if( clock_gettime( CLOCK_MONOTONIC, &ts ) != 0 )
    {
        return 0;
    }

    return int64_t( ts.tv_sec ) * 1000000000 + ts.tv_nsec;
}

inline timespec toTimespec( int64_t ns )
{
    return { ns / 1000000000, ns % 1000000000 };
}

inline bool validRecord( const Record &record, int64_t now )
{
    return validRecordVersion( record, now, layoutVersion );
}

inline bool validRecordVersion( const Record &record, int64_t now, int64_t supportedVersion )
{
    if( record[version] != supportedVersion || record[epoch] <= 0 || record[generation] <= 0 ||
        ( record[state] != acquiring && record[state] != armed ) || record[steps] < 1 || record[steps] > maxSteps ||
        record[period] <= 0 || record[period] > 1000000000 || record[reference] <= 0 || record[reference] > now ||
        record[expires] <= now || record[expires] - record[reference] > 3 * record[period] )
    {
        return false;
    }

    for( int j = 0; j < record[steps]; ++j )
    {
        if( record[cutoffs + j] != 0 &&
            ( record[cutoffs + j] >= record[targets + j] || record[targets + j] > record[expires] ) )
        {
            return false;
        }
    }

    return true;
}

inline int nextStep( const Record &record, int64_t now )
{
    if( !validRecord( record, now ) )
    {
        return -1;
    }

    int selected = -1;

    for( int j = 0; j < record[steps]; ++j )
    {
        if( record[cutoffs + j] > now && ( selected < 0 || record[cutoffs + j] < record[cutoffs + selected] ) )
        {
            selected = j;
        }
    }

    return selected;
}

inline bool validProductionRecord( const Record &record, int64_t now )
{
    return validRecordVersion( record, now, productionVersion );
}

inline int nextProductionStep( const Record &record, int64_t now )
{
    if( !validProductionRecord( record, now ) )
    {
        return -1;
    }

    int selected = -1;

    for( int j = 0; j < record[steps]; ++j )
    {
        if( record[cutoffs + j] > now && ( selected < 0 || record[cutoffs + j] < record[cutoffs + selected] ) )
        {
            selected = j;
        }
    }

    return selected;
}

inline Request request( const Record &record, int step )
{
    return { record[epoch],
             record[generation],
             record[cycle] + ( step == 0 ? 1 : 0 ),
             step,
             record[targets + step],
             record[cutoffs + step] };
}

inline bool validCommand( const CommitRecord &command, int64_t now )
{
    return command[cmdVersion] == productionVersion && command[cmdEpoch] > 0 && command[cmdGeneration] > 0 &&
           command[cmdCycle] >= 0 && command[cmdStep] >= 0 && command[cmdStep] < maxSteps && command[cmdCutoff] > 0 &&
           command[cmdTarget] > command[cmdCutoff] && command[cmdPayloadCounter] > 0 && command[cmdReadyNs] > 0 &&
           command[cmdReadyNs] <= command[cmdCutoff] && command[cmdReadyNs] <= now && command[cmdSession] > 0 &&
           command[cmdSequence] > 0 && command[cmdPublisher] > 0 && command[cmdPayloadInode] > 0;
}

inline bool commandMatches( const CommitRecord &command, const Request &slot )
{
    return command[cmdVersion] == productionVersion && command[cmdEpoch] == slot.epoch &&
           command[cmdGeneration] == slot.generation && command[cmdCycle] == slot.cycle &&
           command[cmdStep] == slot.step && command[cmdTarget] == slot.targetNs && command[cmdCutoff] == slot.cutoffNs;
}

inline bool acknowledges( const AckRecord &ack, const CommitRecord &command )
{
    if( ack[ackStatus] != copied && ack[ackStatus] != discarded )
    {
        return false;
    }

    for( size_t j = 0; j < command.size(); ++j )
    {
        if( ack[j] != command[j] )
        {
            return false;
        }
    }

    return true;
}

} // namespace synco
} // namespace app
} // namespace MagAOX
#endif
