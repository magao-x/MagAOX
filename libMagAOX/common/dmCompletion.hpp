/** \file dmCompletion.hpp
 * \brief Versioned command identity and reader-completion records for DM controllers.
 */
#ifndef common_dmCompletion_hpp
#define common_dmCompletion_hpp

#include <array>
#include <cstddef>
#include <cstdint>

#include "dmTiming.hpp"

namespace MagAOX
{
namespace dmCompletion
{

constexpr uint64_t magic   = 0x4d414758444d444eULL; ///< Distinguish this protocol from unrelated uint64 streams.
constexpr uint64_t version = 1;                     ///< Current fixed-layout protocol version.
constexpr size_t   words   = 16;                    ///< Number of uint64 words in one published record.

/// Reader state; a semaphore notification alone never establishes completion.
enum class State : uint64_t
{
    unavailable,
    busy,
    complete,
    error,
    unqualified
};

/// One controller attempt, identified independently of semaphore delivery.
/** Complete means a successful, genuine callback finished all software channel
 * reads while the combined input identity remained stable. It does not assert
 * physical mirror settling, atomic multi-stream provenance or ownership of any
 * future callback. External publication seconds/nanoseconds are identity only;
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
    int64_t                   callbackResult{ 0 }; ///< Software callback result, distinct from backend qualification.
    uint64_t                  publisherPid{ 0 };   ///< Publisher PID for diagnostics, not lifetime identity.
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

/// Return the diagnostic name for one protocol state.
inline const char *stateName( State state /**< [in] Reader state to describe. */ ) noexcept;

inline Wire encode( const Record &record ) noexcept
{
    return { magic,
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
             0 };
}

inline bool decode( const Wire &wire, Record &record ) noexcept
{
    if( wire[0] != magic || wire[1] != version || wire[5] > static_cast<uint64_t>( State::unqualified ) ||
        wire[12] > static_cast<uint64_t>( MagAOX::dmTiming::Outcome::rejected ) || wire[15] != 0 )
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
    record                 = result;
    return true;
}

inline bool qualified( const Record &record ) noexcept
{
    return record.protocolVersion == version && record.session != 0 && record.epoch != 0 && record.sequence != 0 &&
           record.state == State::complete && record.inputInode != 0 && record.inputCounter != 0 &&
           record.publicationSec > 0 && record.publicationNsec >= 0 && record.publicationNsec < 1000000000 &&
           record.startedNs > 0 && record.finishedNs >= record.startedNs && record.callbackResult == 0 &&
           record.outcome == MagAOX::dmTiming::Outcome::success;
}

inline const char *stateName( State state ) noexcept
{
    constexpr const char *names[] = { "unavailable", "busy", "complete", "error", "unqualified" };
    const auto            index   = static_cast<size_t>( state );
    return index < sizeof( names ) / sizeof( names[0] ) ? names[index] : "invalid";
}

} // namespace dmCompletion
} // namespace MagAOX
#endif
