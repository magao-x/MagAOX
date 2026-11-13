/** \file dmCompletion.hpp
 * \brief Versioned command identity and reader-completion records for DM controllers.
 */
#ifndef common_dmCompletion_hpp
#define common_dmCompletion_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "dmTiming.hpp"

namespace MagAOX
{
namespace dmCompletion
{

constexpr uint64_t magic        = 0x4d414758444d444eULL; ///< Distinguish this protocol from unrelated uint64 streams.
constexpr uint64_t readersMagic = 0x4d414758444d5244ULL; ///< Distinguish early read release from full completion.
constexpr uint64_t sdkMagic     = 0x4d414758444d5344ULL; ///< Distinguish SDK return from reader/full completion.
constexpr uint64_t version      = 1;                     ///< Current fixed-layout protocol version.
constexpr size_t   words        = 16;                    ///< Number of uint64 words in one published record.
constexpr int64_t  unavailableResult = std::numeric_limits<int64_t>::min(); ///< No actual backend result was supplied.

/// Separate endpoint contracts; a read release never implies SDK success or backend readiness.
enum class Kind : uint8_t
{
    completion,
    readers,
    sdk
};

/// Reader state; a semaphore notification alone never establishes completion.
enum class State : uint64_t
{
    unavailable,
    busy,
    complete,
    error,
    unqualified,
    released
};

/// One controller attempt, identified independently of semaphore delivery.
/** Completion-kind complete means a successful genuine job finished its channel
 * reads and required software products using a stable consumed input identity.
 * Legacy publication rechecks the live input; detached publication relies on
 * caller validation of the frozen job before its ownership transfer.
 * Reader-kind released means external reads finished with a stable identity;
 * its SDK outcome may still be unknown or may later fail. Neither asserts
 * physical mirror settling, atomic multi-stream provenance or ownership of any
 * future callback. SDK-kind complete reports only successful SDK return for
 * its frozen command; it does not assert output publication or reader clearance.
 * External publication seconds/nanoseconds are identity only;
 * the started/finished timestamps use the local monotonic clock.
 */
struct Record
{
    uint64_t                  protocolVersion{ version }; ///< Serialization version.
    uint64_t                  session{ 0 };               ///< Nonzero publisher startup nonce; changes on restart.
    uint64_t                  epoch{ 0 };    ///< Attachment or counter-reset generation within the session.
    uint64_t                  sequence{ 0 }; ///< Callback attempt, including unqualified attempts.
    State                     state{ State::unavailable }; ///< Current reader lifecycle state.
    uint64_t                  inputInode{ 0 };             ///< Combined input stream's metadata inode.
    uint64_t                  inputCounter{ 0 };           ///< Stable input cnt0 associated with this attempt.
    int64_t                   publicationSec{ 0 };  ///< Raw input writetime seconds; never mixed with monotonic time.
    int64_t                   publicationNsec{ 0 }; ///< Raw input writetime nanoseconds.
    int64_t                   startedNs{ 0 };       ///< Local monotonic callback entry time.
    int64_t                   finishedNs{ 0 };      ///< Local monotonic completion time, zero while busy.
    MagAOX::dmTiming::Outcome outcome{ MagAOX::dmTiming::Outcome::unknown }; ///< Qualified backend result.
    int64_t  callbackResult{ 0 };      ///< Final callback result, or frozen reader-stage result for early release.
    uint64_t publisherPid{ 0 };        ///< Publisher PID for diagnostics, not lifetime identity.
    Kind     kind{ Kind::completion }; ///< Distinct wire protocol and endpoint qualification contract.
    int64_t  backendResult{
        unavailableResult }; ///< Exact SDK result on reader-release or SDK protocols, independently of release.
};

/// Fixed word transport, independent of struct padding and enum representation.
using Wire = std::array<uint64_t, words>;

/// Encode one record without copying object padding.
inline Wire encode( const Record &record /**< [in] Local record to serialize. */ ) noexcept;

/// Decode a recognized record; leave the destination unchanged on invalid protocol values.
inline bool decode( const Wire &wire /**< [in] Complete transport words. */,
                    Record     &record /**< [out] Decoded record when valid. */ ) noexcept;

/// Return whether this record explicitly qualifies successful software reader completion.
inline bool qualified( const Record &record /**< [in] Coherently observed record. */ ) noexcept;

/// Qualify finished external reads independently of a pending or failed SDK submission.
inline bool readersQualified( const Record &record /**< [in] Coherently observed early-release record. */ ) noexcept;

/// Qualify SDK return independently of reader release and full software completion.
inline bool sdkQualified( const Record &record /**< [in] Coherently observed SDK-result record. */ ) noexcept;

/// Return whether the frozen attempt has representable identity and endpoint timestamps.
inline bool validIdentity( const Record &record /**< [in] Coherently observed endpoint record. */ ) noexcept;

/// Return the diagnostic name for one protocol state.
inline const char *stateName( State state /**< [in] Reader state to describe. */ ) noexcept;

inline Wire encode( const Record &record ) noexcept
{
    return { record.kind == Kind::completion
                 ? magic
                 : ( record.kind == Kind::readers ? readersMagic : ( record.kind == Kind::sdk ? sdkMagic : 0 ) ),
             record.protocolVersion,
             record.session,
             record.epoch,
             record.sequence,
             static_cast<uint64_t>( record.state ),
             record.inputInode,
             record.inputCounter,
             static_cast<uint64_t>( record.publicationSec ),
             static_cast<uint64_t>( record.publicationNsec ),
             static_cast<uint64_t>( record.startedNs ),
             static_cast<uint64_t>( record.finishedNs ),
             static_cast<uint64_t>( record.outcome ),
             static_cast<uint64_t>( record.callbackResult ),
             record.publisherPid,
             record.kind == Kind::readers || record.kind == Kind::sdk ? static_cast<uint64_t>( record.backendResult )
                                                                      : 0 };
}

inline bool decode( const Wire &wire, Record &record ) noexcept
{
    const bool readers = wire[0] == readersMagic;
    const bool sdk     = wire[0] == sdkMagic;

    if( ( wire[0] != magic && !readers && !sdk ) || wire[1] != version ||
        wire[5] > static_cast<uint64_t>( readers ? State::released : State::unqualified ) ||
        wire[12] > static_cast<uint64_t>( MagAOX::dmTiming::Outcome::rejected ) ||
        ( !readers && !sdk && wire[15] != 0 ) || ( readers && wire[5] == static_cast<uint64_t>( State::complete ) ) )
    {
        return false;
    }

    Record result;
    result.protocolVersion = wire[1];
    result.session         = wire[2];
    result.epoch           = wire[3];
    result.sequence        = wire[4];
    result.state           = static_cast<State>( wire[5] );
    result.inputInode      = wire[6];
    result.inputCounter    = wire[7];
    result.publicationSec  = static_cast<int64_t>( wire[8] );
    result.publicationNsec = static_cast<int64_t>( wire[9] );
    result.startedNs       = static_cast<int64_t>( wire[10] );
    result.finishedNs      = static_cast<int64_t>( wire[11] );
    result.outcome         = static_cast<MagAOX::dmTiming::Outcome>( wire[12] );
    result.callbackResult  = static_cast<int64_t>( wire[13] );
    result.publisherPid    = wire[14];
    result.kind            = readers ? Kind::readers : ( sdk ? Kind::sdk : Kind::completion );
    result.backendResult   = readers || sdk ? static_cast<int64_t>( wire[15] ) : unavailableResult;
    record                 = result;
    return true;
}

inline bool qualified( const Record &record ) noexcept
{
    return record.kind == Kind::completion && record.state == State::complete && validIdentity( record ) &&
           record.callbackResult == 0 && record.outcome == MagAOX::dmTiming::Outcome::success;
}

inline bool readersQualified( const Record &record ) noexcept
{
    return record.kind == Kind::readers && record.state == State::released && validIdentity( record ) &&
           record.callbackResult == 0;
}

inline bool sdkQualified( const Record &record ) noexcept
{
    return record.kind == Kind::sdk && record.state == State::complete && validIdentity( record ) &&
           record.callbackResult == 0 && record.outcome == MagAOX::dmTiming::Outcome::success;
}

inline bool validIdentity( const Record &record ) noexcept
{
    return record.protocolVersion == version && record.session != 0 && record.epoch != 0 && record.sequence != 0 &&
           record.inputInode != 0 && record.inputCounter != 0 && record.publicationSec > 0 &&
           record.publicationNsec >= 0 && record.publicationNsec < 1000000000 && record.startedNs > 0 &&
           record.finishedNs >= record.startedNs;
}

inline const char *stateName( State state ) noexcept
{
    constexpr const char *names[] = { "unavailable", "busy", "complete", "error", "unqualified", "released" };
    const auto            index   = static_cast<size_t>( state );
    return index < sizeof( names ) / sizeof( names[0] ) ? names[index] : "invalid";
}

} // namespace dmCompletion
} // namespace MagAOX
#endif
