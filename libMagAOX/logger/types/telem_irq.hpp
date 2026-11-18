/** \file telem_irq.hpp
 * \brief Compact IRQ policy flags and backward-compatible placement-record decoding.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_irq_hpp
#define logger_types_telem_irq_hpp

#include <algorithm>

#include <mx/math/floatUtils.hpp>

#include "generated/telem_irq_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Compact policy results in the group order recorded in startup logs.
/** \ingroup logger_types */
struct telem_irq : public flatbuffer_log
{
    /// Unique logger event code.
    static constexpr flatlogs::eventCodeT eventCode = eventCodes::TELEM_IRQ;

    /// Default telemetry priority.
    static constexpr flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    /// Most recent telemetry record time used by the scheduler.
    static timespec lastRecord;

    /// Requested and effective routing observed for one current IRQ number.
    struct irqT
    {
        unsigned    m_number{ 0 }; ///< Current IRQ number, rediscovered at each check.
        std::string m_requested;   ///< Canonical allowed CPU list, or unreadable/malformed input.
        std::string m_effective;   ///< Canonical effective CPU list, empty if unavailable.

        /// Compare routing state, including the current IRQ identity.
        bool operator==( const irqT &other /**< [in] snapshot to compare */ ) const;
    };

    /// One logical device group and all its resolved IRQ vectors.
    struct groupT
    {
        std::string       m_name;            ///< Stable configuration name identifying the device group.
        std::string       m_expected;        ///< Expected normalized CPU list, empty when unavailable.
        std::string       m_status;          ///< Verification result and failure reasons.
        bool              m_warning{ true }; ///< Whether operator attention is required by this group's policy.
        double            m_checkTime{ 0 };  ///< Unix time of the actual check, retained on forced cached records.
        std::vector<irqT> m_irqs;            ///< Observed IRQs in numeric order.

        /// Compare diagnostic state, deliberately excluding the collection timestamp.
        bool operator==( const groupT &other /**< [in] snapshot to compare */ ) const;
    };

    /// Serialized policy flags and actual collection time, without repeated placement metadata.
    struct messageT : public fbMessage
    {
        /// Serialize zero for unchecked/unacceptable groups and one for completed acceptable checks.
        messageT( const std::vector<uint8_t> &ok /**< [in] policy flags in startup-logged group order */,
                  double checkTime /**< [in] Unix time of the actual check, or zero before any check */ );
    };

    /// Verify one compact or legacy representation, including compact flags and collection time.
    static bool verify( flatlogs::bufferPtrT &logBuff /**< [in] complete log buffer */,
                        flatlogs::msgLenT     len /**< [in] payload length */ );

    /// Format compact flags and check time, or original details from a legacy record.
    static std::string msgString( void             *msgBuffer /**< [in] serialized payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );

    /// Get zero/one policy flags, converting legacy group results when necessary.
    static std::vector<unsigned> ok( void *msgBuffer /**< [in] serialized payload */ );

    /// Get the actual check time, or the latest group check time in a legacy record.
    static double checkTime( void *msgBuffer /**< [in] serialized payload */ );

    /// Decode legacy identified snapshots, or return an empty vector for compact records.
    static std::vector<groupT> groups( void *msgBuffer /**< [in] serialized payload */ );

    /// Get legacy logical group names, or an empty vector for compact records.
    static std::vector<std::string> names( void *msgBuffer /**< [in] serialized payload */ );

    /// Get legacy verification results in names() order, or an empty vector for compact records.
    static std::vector<std::string> statuses( void *msgBuffer /**< [in] serialized payload */ );

    /// Get legacy expected CPU lists in names() order, or an empty vector for compact records.
    static std::vector<std::string> expected( void *msgBuffer /**< [in] serialized payload */ );

    /// Get each legacy group time, or repeat the compact check time once per policy flag.
    static std::vector<double> checkTimes( void *msgBuffer /**< [in] serialized payload */ );

    /// Get whether any compact flag is zero, or any legacy group reported a warning.
    static bool warning( void *msgBuffer /**< [in] serialized payload */ );

    /// Get metadata for a supported field, or an empty descriptor for an unknown field.
    static logMetaDetail getAccessor( const std::string &member /**< [in] field name */ );
};

inline bool telem_irq::irqT::operator==( const irqT &other ) const
{
    return m_number == other.m_number && m_requested == other.m_requested && m_effective == other.m_effective;
}

inline bool telem_irq::groupT::operator==( const groupT &other ) const
{
    return m_name == other.m_name && m_expected == other.m_expected && m_status == other.m_status &&
           m_warning == other.m_warning && m_irqs == other.m_irqs;
}

inline telem_irq::messageT::messageT( const std::vector<uint8_t> &ok, double checkTime )
{
    auto flags = builder.CreateVector( ok );
    builder.Finish( CreateTelem_irq_fb( builder, 0, flags, checkTime ) );
}

inline bool telem_irq::verify( flatlogs::bufferPtrT &logBuff, flatlogs::msgLenT len )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ), len );

    if( !VerifyTelem_irq_fbBuffer( verifier ) )
    {
        return false;
    }

    const auto *record = GetTelem_irq_fb( flatlogs::logHeader::messageBuffer( logBuff ) );
    const auto *flags  = record->ok();

    if( ( flags != nullptr ) == ( record->groups() != nullptr ) )
    {
        return false;
    }

    if( flags )
    {
        if( !mx::math::isFinite( record->check_time() ) || record->check_time() < 0 )
        {
            return false;
        }

        for( auto value : *flags )
        {
            if( value > 1 )
            {
                return false;
            }
        }
    }

    return true;
}

inline std::string telem_irq::msgString( void *msgBuffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    std::string text   = "[irq]";
    const auto *record = GetTelem_irq_fb( msgBuffer );

    if( record->ok() )
    {
        text += " ok=";
        bool first = true;

        for( auto value : *record->ok() )
        {
            if( !first )
            {
                text += ',';
            }

            text += std::to_string( value );
            first = false;
        }

        return text + " checked=" + std::to_string( record->check_time() );
    }

    if( !record->groups() )
    {
        return text;
    }

    for( const auto *group : *record->groups() )
    {
        text += " " + group->name()->str() + " expected=" + group->expected()->str() +
                " status=" + group->status()->str() + " warning=" + std::to_string( group->warning() ) +
                " checked=" + std::to_string( group->check_time() );

        for( const auto *irq : *group->irqs() )
        {
            text += " IRQ" + std::to_string( irq->number() ) + " requested=" + irq->requested()->str() +
                    " effective=" + irq->effective()->str();
        }
    }

    return text;
}

inline std::vector<unsigned> telem_irq::ok( void *msgBuffer )
{
    std::vector<unsigned> values;
    const auto           *record = GetTelem_irq_fb( msgBuffer );

    if( record->ok() )
    {
        return { record->ok()->begin(), record->ok()->end() };
    }

    if( record->groups() )
    {
        for( const auto *group : *record->groups() )
        {
            values.push_back( !group->warning() && group->status()->str() != "unchecked" );
        }
    }

    return values;
}

inline double telem_irq::checkTime( void *msgBuffer )
{
    const auto *record = GetTelem_irq_fb( msgBuffer );

    if( record->ok() )
    {
        return record->check_time();
    }

    double latest = 0;

    if( record->groups() )
    {
        for( const auto *group : *record->groups() )
        {
            latest = std::max( latest, group->check_time() );
        }
    }

    return latest;
}

inline std::vector<telem_irq::groupT> telem_irq::groups( void *msgBuffer )
{
    std::vector<groupT> values;
    const auto         *legacy = GetTelem_irq_fb( msgBuffer )->groups();

    if( !legacy )
    {
        return values;
    }

    for( const auto *record : *legacy )
    {
        groupT group;
        group.m_name      = record->name()->str();
        group.m_expected  = record->expected()->str();
        group.m_status    = record->status()->str();
        group.m_warning   = record->warning();
        group.m_checkTime = record->check_time();

        for( const auto *irq : *record->irqs() )
        {
            group.m_irqs.push_back( { irq->number(), irq->requested()->str(), irq->effective()->str() } );
        }

        values.push_back( std::move( group ) );
    }

    return values;
}

inline std::vector<std::string> telem_irq::names( void *msgBuffer )
{
    std::vector<std::string> values;
    const auto              *legacy = GetTelem_irq_fb( msgBuffer )->groups();

    if( !legacy )
    {
        return values;
    }

    for( const auto *group : *legacy )
    {
        values.push_back( group->name()->str() );
    }

    return values;
}

inline std::vector<std::string> telem_irq::statuses( void *msgBuffer )
{
    std::vector<std::string> values;
    const auto              *legacy = GetTelem_irq_fb( msgBuffer )->groups();

    if( !legacy )
    {
        return values;
    }

    for( const auto *group : *legacy )
    {
        values.push_back( group->status()->str() );
    }

    return values;
}

inline std::vector<std::string> telem_irq::expected( void *msgBuffer )
{
    std::vector<std::string> values;
    const auto              *legacy = GetTelem_irq_fb( msgBuffer )->groups();

    if( !legacy )
    {
        return values;
    }

    for( const auto *group : *legacy )
    {
        values.push_back( group->expected()->str() );
    }

    return values;
}

inline std::vector<double> telem_irq::checkTimes( void *msgBuffer )
{
    std::vector<double> values;
    const auto         *record = GetTelem_irq_fb( msgBuffer );

    if( record->ok() )
    {
        return std::vector<double>( record->ok()->size(), record->check_time() );
    }

    if( !record->groups() )
    {
        return values;
    }

    for( const auto *group : *record->groups() )
    {
        values.push_back( group->check_time() );
    }

    return values;
}

inline bool telem_irq::warning( void *msgBuffer )
{
    const auto *record = GetTelem_irq_fb( msgBuffer );

    if( record->ok() )
    {
        for( auto value : *record->ok() )
        {
            if( !value )
            {
                return true;
            }
        }

        return false;
    }

    if( !record->groups() )
    {
        return false;
    }

    for( const auto *group : *record->groups() )
    {
        if( group->warning() )
        {
            return true;
        }
    }

    return false;
}

inline logMetaDetail telem_irq::getAccessor( const std::string &member )
{
    if( member == "ok" )
    {
        return { "IRQ OK",
                 "policy flags in startup-logged group order",
                 logMeta::valTypes::Vector_UInt,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &ok ),
                 true };
    }

    if( member == "checkTime" )
    {
        return { "IRQ CHECK TIME",
                 "actual IRQ check time [Unix seconds]",
                 logMeta::valTypes::Double,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &checkTime ),
                 true };
    }

    if( member == "names" )
    {
        return { "IRQ NAMES",
                 "logical IRQ group names",
                 logMeta::valTypes::Vector_String,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &names ),
                 true };
    }

    if( member == "statuses" )
    {
        return { "IRQ STATUS",
                 "IRQ placement verification status",
                 logMeta::valTypes::Vector_String,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &statuses ),
                 true };
    }

    if( member == "expected" )
    {
        return { "IRQ EXPECTED",
                 "expected IRQ CPU lists",
                 logMeta::valTypes::Vector_String,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &expected ),
                 true };
    }

    if( member == "checkTimes" )
    {
        return { "IRQ CHECK TIMES",
                 "actual IRQ check times [Unix seconds]",
                 logMeta::valTypes::Vector_Double,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &checkTimes ),
                 true };
    }

    if( member == "warning" )
    {
        return { "IRQ WARNING",
                 "whether any IRQ group requires operator attention",
                 logMeta::valTypes::Bool,
                 logMeta::metaTypes::State,
                 reinterpret_cast<void *>( &warning ),
                 true };
    }

    return {};
}

} // namespace logger
} // namespace MagAOX
#endif // logger_types_telem_irq_hpp
