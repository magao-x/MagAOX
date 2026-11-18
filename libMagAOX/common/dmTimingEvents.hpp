/** \file dmTimingEvents.hpp
 * \brief Fixed records joining native SDK entries to qualified syncopation main identities.
 */
#ifndef common_dmTimingEvents_hpp
#define common_dmTimingEvents_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace MagAOX
{
namespace dmTimingEvents
{

/// Distinct native endpoint, qualified main classification and operating-mode observations.
enum class Kind : uint64_t
{
    sdk  = 1,
    main = 2,
    mode = 3
};

/// Exact combined-stream publication identity; its timestamp is identity, not a monotonic endpoint.
struct Key
{
    uint64_t inode{ 0 };   ///< Combined publisher incarnation from image metadata.
    uint64_t counter{ 0 }; ///< Combined publication counter within this incarnation.
    int64_t  sec{ 0 };     ///< Raw producer writetime seconds.
    int64_t  nsec{ 0 };    ///< Raw producer writetime fractional nanoseconds.
};

/// One producer observation, independent of ring publication or downstream arrival order.
struct Record
{
    Kind     kind{ Kind::sdk };   ///< Select the endpoint/classification/mode contract.
    Key      key;                 ///< Exact combined key; empty for mode records.
    uint64_t sequence{ 0 };       ///< Producer sequence including observations dropped before transport publication.
    uint64_t generation{ 0 };     ///< Synco classification/mode generation; zero is allowed for SDK records.
    uint64_t primaryInode{ 0 };   ///< Genuine primary stream incarnation; required for main records.
    uint64_t primaryCounter{ 0 }; ///< Genuine primary cycle counter; required for main records.
    uint64_t steps{ 0 };          ///< Configured main-plus-auxiliary count; SDK records may leave this zero.
    bool     enabled{ false };    ///< Synco mode at classification; mains are labelled while enabled or disabled.
    int64_t  startNs{ 0 };        ///< Original MONOTONIC SDK entry; zero for classification and mode records.
};

constexpr size_t   words    = 12; ///< Fixed uint64 transport size, independent of structure padding.
constexpr uint64_t maxSteps = 16; ///< Maximum configured scheduler subdivision supported by this protocol.

/// Fixed word layout carried atomically by each ring slot.
using Wire = std::array<uint64_t, words>;

/// Require a positive, representable exact combined publication identity.
inline bool valid( const Key &key /**< [in] Combined publication identity to validate. */ ) noexcept;

/// Compare all four identity components without converting their clock.
inline bool equal( const Key &first /**< [in] First combined identity. */,
                   const Key &second /**< [in] Second combined identity. */ ) noexcept;

/// Validate the required fields for one endpoint, main classification or mode record.
inline bool valid( const Record &record /**< [in] Producer observation to validate. */ ) noexcept;

/// Encode one record without copying enum or structure padding.
inline Wire encode( const Record &record /**< [in] Producer observation to encode. */ ) noexcept;

/// Decode only recognized and valid records, preserving the destination on failure.
inline bool decode( const Wire &wire /**< [in] Complete transport words. */,
                    Record     &record /**< [out] Decoded observation on success. */ ) noexcept;

inline bool valid( const Key &key ) noexcept
{
    return key.inode && key.counter && key.sec > 0 && key.nsec >= 0 && key.nsec < 1000000000 &&
           key.sec <= ( std::numeric_limits<int64_t>::max() - key.nsec ) / 1000000000;
}

inline bool equal( const Key &first, const Key &second ) noexcept
{
    return first.inode == second.inode && first.counter == second.counter && first.sec == second.sec &&
           first.nsec == second.nsec;
}

inline bool valid( const Record &record ) noexcept
{
    if( !record.sequence )
    {
        return false;
    }

    switch( record.kind )
    {
    case Kind::sdk:
        return valid( record.key ) && record.startNs > 0;
    case Kind::main:
        return valid( record.key ) && record.generation && record.primaryInode && record.primaryCounter &&
               record.steps > 0 && record.steps <= maxSteps && record.startNs == 0;
    case Kind::mode:
        return equal( record.key, Key{} ) && record.generation && record.steps <= maxSteps &&
               ( !record.enabled || record.steps > 0 ) && record.startNs == 0;
    }

    return false;
}

inline Wire encode( const Record &record ) noexcept
{
    return { static_cast<uint64_t>( record.kind ),
             record.key.inode,
             record.key.counter,
             static_cast<uint64_t>( record.key.sec ),
             static_cast<uint64_t>( record.key.nsec ),
             record.sequence,
             record.generation,
             record.primaryInode,
             record.primaryCounter,
             record.steps,
             uint64_t( record.enabled ),
             static_cast<uint64_t>( record.startNs ) };
}

inline bool decode( const Wire &wire, Record &record ) noexcept
{
    if( wire[0] < uint64_t( Kind::sdk ) || wire[0] > uint64_t( Kind::mode ) || wire[10] > 1 )
    {
        return false;
    }

    Record value;
    value.kind           = static_cast<Kind>( wire[0] );
    value.key            = { wire[1], wire[2], static_cast<int64_t>( wire[3] ), static_cast<int64_t>( wire[4] ) };
    value.sequence       = wire[5];
    value.generation     = wire[6];
    value.primaryInode   = wire[7];
    value.primaryCounter = wire[8];
    value.steps          = wire[9];
    value.enabled        = wire[10] != 0;
    value.startNs        = static_cast<int64_t>( wire[11] );

    if( !valid( value ) )
    {
        return false;
    }

    record = value;
    return true;
}

} // namespace dmTimingEvents
} // namespace MagAOX
#endif // common_dmTimingEvents_hpp
