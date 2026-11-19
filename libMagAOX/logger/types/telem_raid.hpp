/** \file telem_raid.hpp
 * \brief Compact indexed software RAID health with explicit validity and collection state.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_raid_hpp
#define logger_types_telem_raid_hpp

#include <cstdint>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <mx/math/floatUtils.hpp>

#include "generated/telem_raid_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Software RAID health keyed by the array indices recorded in discovery logs.
/** \ingroup logger_types */
struct telem_raid : public flatbuffer_log
{
    /// Logger dispatch code.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_RAID;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Most recent telemetry record time maintained by the telemetry scheduler.
    static timespec lastRecord;

    /// No assigned discovery index; not permitted in serialized array records.
    static constexpr uint32_t UnknownIndex = std::numeric_limits<uint32_t>::max();

    /// Independent health facts; only current usable assessments set Valid.
    enum flagBits : uint32_t
    {
        Valid     = 1U << 0, ///< The current array health assessment is usable.
        OK        = 1U << 1, ///< The array satisfies the monitoring policy.
        Degraded  = 1U << 2, ///< A redundant array has fewer active members than expected.
        Missing   = 1U << 3, ///< A known array is absent, or its mdstat source has disappeared.
        Inactive  = 1U << 4, ///< The array is present but inactive.
        Redundant = 1U << 5, ///< Expected and active member counts describe a redundant array.
        Container = 1U << 6, ///< A recognized metadata container; failed components still clear OK.
        Faulty    = 1U << 7, ///< A failed component (including a spare) or broken array is reported.
        AllFlags  = 255U     ///< All defined array flags.
    };

    /// Overall collection outcome, independent of individual array health.
    enum collectionStatus : uint8_t
    {
        Unchecked  = 0, ///< No collection attempt has completed.
        Ready      = 1, ///< A complete snapshot was read and parsed.
        ReadError  = 2, ///< The snapshot could not be read.
        ParseError = 3, ///< The snapshot was malformed or incomplete.
        Disabled   = 4, ///< Monitoring is disabled by configuration.
        NotPresent = 5  ///< The snapshot file does not exist on this host.
    };

    /// One array's numeric policy result, without repeated names or member details.
    struct arrayT
    {
        uint32_t m_index{ UnknownIndex }; ///< Stable index assigned and logged on discovery.
        uint32_t m_flags{ 0 };            ///< Current health facts from flagBits.
        uint32_t m_expected{ 0 };         ///< Expected redundant members, or zero without Redundant.
        uint32_t m_active{ 0 };           ///< Active redundant members, or zero without Redundant.

        /// Compare policy results for change-triggered recording.
        bool operator==( const arrayT &other /**< [in] array state to compare */ ) const;
    };

    /// Serialized collection outcome, numeric array health, and actual check time.
    struct messageT : public fbMessage
    {
        /// Serialize numeric state without repeated discovery metadata.
        messageT( uint8_t                    status /**< [in] overall collectionStatus */,
                  const std::vector<arrayT> &arrays /**< [in] array results in discovery-index order */,
                  double checkTime /**< [in] Unix UTC seconds of the attempt, or zero before collection */ );
    };

    /// Verify the payload structure, collection state, time, indices, counts, and health flags.
    static bool verify( flatlogs::bufferPtrT &buffer /**< [in] complete log entry */,
                        flatlogs::msgLenT     len /**< [in] serialized payload length */ );

    /// Format indexed health facts and the actual check time for operator-readable logs.
    static std::string msgString( void             *buffer /**< [in] serialized payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );

    /// Decode each numeric array result in stored order.
    static std::vector<arrayT> arrays( void *buffer /**< [in] serialized payload */ );

    /// Get the overall collectionStatus value.
    static unsigned status( void *buffer /**< [in] serialized payload */ );

    /// Get the actual collection attempt time in Unix UTC seconds.
    static double checkTime( void *buffer /**< [in] serialized payload */ );

    /// Get whether each array has a current usable health assessment.
    static std::vector<unsigned> valid( void *buffer /**< [in] serialized payload */ );

    /// Get whether each array satisfies the monitoring policy.
    static std::vector<unsigned> ok( void *buffer /**< [in] serialized payload */ );

    /// Get whether each array has lost required redundant members.
    static std::vector<unsigned> degraded( void *buffer /**< [in] serialized payload */ );

    /// Get metadata accessors; all vectors share the stored array order.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );

  private:
    /// Extract one numeric array field in stored order.
    template <auto member>
    static std::vector<unsigned> numbers( void *buffer /**< [in] serialized payload */ );

    /// Extract a zero/one flag vector in stored order.
    template <uint32_t flag>
    static std::vector<unsigned> flags( void *buffer /**< [in] serialized payload */ );
};

inline bool telem_raid::arrayT::operator==( const arrayT &other ) const
{
    return std::tie( m_index, m_flags, m_expected, m_active ) ==
           std::tie( other.m_index, other.m_flags, other.m_expected, other.m_active );
}

inline telem_raid::messageT::messageT( uint8_t status, const std::vector<arrayT> &arrays, double checkTime )
{
    std::vector<Raid_array_fb> records;
    records.reserve( arrays.size() );

    for( const auto &array : arrays )
    {
        records.emplace_back( array.m_index, array.m_flags, array.m_expected, array.m_active );
    }

    const auto samples = builder.CreateVectorOfStructs( records );
    builder.Finish( CreateTelem_raid_fb( builder, status, samples, checkTime ) );
}

inline bool telem_raid::verify( flatlogs::bufferPtrT &buffer, flatlogs::msgLenT len )
{
    auto                 *payload = static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( buffer ) );
    flatbuffers::Verifier verifier( payload, len );

    if( !VerifyTelem_raid_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *record = GetTelem_raid_fb( payload );

    if( record->status() > NotPresent || !mx::math::isFinite( record->check_time() ) || record->check_time() < 0 )
    {
        return false;
    }

    std::set<uint32_t> indices;

    for( const auto *array : *record->arrays() )
    {
        const auto mask = array->flags();

        if( array->index() == UnknownIndex || !indices.insert( array->index() ).second || ( mask & ~AllFlags ) )
        {
            return false;
        }

        if( !( mask & Valid ) )
        {
            if( ( mask != 0 && mask != Missing ) || array->expected() != 0 || array->active() != 0 )
            {
                return false;
            }

            continue;
        }

        if( ( mask & Missing ) || ( ( mask & OK ) && ( mask & ( Degraded | Inactive | Faulty ) ) ) ||
            ( ( mask & Container ) && mask != ( Valid | OK | Container ) && mask != ( Valid | Container | Faulty ) ) )
        {
            return false;
        }

        if( mask & Redundant )
        {
            if( array->expected() == 0 || array->active() > array->expected() ||
                bool( mask & Degraded ) != ( array->active() < array->expected() ) )
            {
                return false;
            }
        }
        else if( ( mask & Degraded ) || array->expected() != 0 || array->active() != 0 )
        {
            return false;
        }
    }

    return true;
}

inline unsigned telem_raid::status( void *buffer )
{
    return GetTelem_raid_fb( buffer )->status();
}

inline double telem_raid::checkTime( void *buffer )
{
    return GetTelem_raid_fb( buffer )->check_time();
}

inline std::vector<telem_raid::arrayT> telem_raid::arrays( void *buffer )
{
    std::vector<arrayT> result;
    const auto         *samples = GetTelem_raid_fb( buffer )->arrays();
    result.reserve( samples->size() );

    for( const auto *array : *samples )
    {
        result.push_back( { array->index(), array->flags(), array->expected(), array->active() } );
    }

    return result;
}

template <auto member>
inline std::vector<unsigned> telem_raid::numbers( void *buffer )
{
    std::vector<unsigned> result;

    for( const auto *array : *GetTelem_raid_fb( buffer )->arrays() )
    {
        result.push_back( ( array->*member )() );
    }

    return result;
}

template <uint32_t flag>
inline std::vector<unsigned> telem_raid::flags( void *buffer )
{
    std::vector<unsigned> result;

    for( const auto *array : *GetTelem_raid_fb( buffer )->arrays() )
    {
        result.push_back( bool( array->flags() & flag ) );
    }

    return result;
}

inline std::vector<unsigned> telem_raid::valid( void *buffer )
{
    return flags<Valid>( buffer );
}

inline std::vector<unsigned> telem_raid::ok( void *buffer )
{
    return flags<OK>( buffer );
}

inline std::vector<unsigned> telem_raid::degraded( void *buffer )
{
    return flags<Degraded>( buffer );
}

inline std::string telem_raid::msgString( void *buffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    std::ostringstream text;
    text << "[raid] status=" << status( buffer ) << " checked=" << std::setprecision( 17 ) << checkTime( buffer );

    for( const auto &array : arrays( buffer ) )
    {
        text << " [index=" << array.m_index << " flags=" << array.m_flags << " active=" << array.m_active << "/"
             << array.m_expected << "]";
    }

    return text.str();
}

inline logMetaDetail telem_raid::getAccessor( const std::string &member )
{
    if( member == "status" )
    {
        return { "RAID STATUS",
                 "RAID collection status code",
                 logMeta::UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &status ),
                 true };
    }

    if( member == "checkTime" )
    {
        return { "RAID CHECK TIME",
                 "actual collection attempt [Unix UTC s]",
                 logMeta::Double,
                 logMeta::State,
                 reinterpret_cast<void *>( &checkTime ),
                 true };
    }

    if( member == "index" )
    {
        return { "RAID INDEX",
                 "array indices in discovery logs",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<&Raid_array_fb::index> ),
                 true };
    }

    if( member == "flags" )
    {
        return { "RAID FLAGS",
                 "current array health bit masks",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<&Raid_array_fb::flags> ),
                 true };
    }

    if( member == "expected" )
    {
        return { "RAID EXPECTED",
                 "expected members; requires Redundant flag",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<&Raid_array_fb::expected> ),
                 true };
    }

    if( member == "active" )
    {
        return { "RAID ACTIVE",
                 "active members; requires Redundant flag",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &numbers<&Raid_array_fb::active> ),
                 true };
    }

    if( member == "valid" )
    {
        return { "RAID VALID",
                 "current usable health assessments",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &valid ),
                 true };
    }

    if( member == "ok" )
    {
        return { "RAID OK",
                 "arrays satisfying the monitoring policy",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &ok ),
                 true };
    }

    if( member == "degraded" )
    {
        return { "RAID DEGRADED",
                 "arrays with missing required redundant members",
                 logMeta::Vector_UInt,
                 logMeta::State,
                 reinterpret_cast<void *>( &degraded ),
                 true };
    }

    return {};
}

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_raid_hpp
