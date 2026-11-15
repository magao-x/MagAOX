/** \file telem_irq.hpp
 * \brief Identified IRQ placement telemetry with explicit verification status.
 * \ingroup logger_types_files
 */
#ifndef logger_types_telem_irq_hpp
#define logger_types_telem_irq_hpp

#include "generated/telem_irq_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Placement snapshots for configured logical IRQ groups.
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

    /// Serialized snapshot input.
    struct messageT : public fbMessage
    {
        /// Serialize all groups and their actual check times.
        messageT( const std::vector<groupT> &groups /**< [in] identified placement snapshots */ );
    };

    /// Verify the complete serialized payload.
    static bool verify( flatlogs::bufferPtrT &logBuff /**< [in] complete log buffer */,
                        flatlogs::msgLenT     len /**< [in] payload length */ );

    /// Format group identity, verification status and observed CPU lists.
    static std::string msgString( void             *msgBuffer /**< [in] serialized payload */,
                                  flatlogs::msgLenT len /**< [in] unused payload length */ );

    /// Decode the complete identified snapshot, including acquisition timestamps.
    static std::vector<groupT> groups( void *msgBuffer /**< [in] serialized payload */ );

    /// Get logical group names in the recorded order.
    static std::vector<std::string> names( void *msgBuffer /**< [in] serialized payload */ );

    /// Get verification results in the same order as names().
    static std::vector<std::string> statuses( void *msgBuffer /**< [in] serialized payload */ );

    /// Get expected CPU lists in the same order as names().
    static std::vector<std::string> expected( void *msgBuffer /**< [in] serialized payload */ );

    /// Get actual check times in the same order as names().
    static std::vector<double> checkTimes( void *msgBuffer /**< [in] serialized payload */ );

    /// Get whether any recorded group requires operator attention.
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

inline telem_irq::messageT::messageT( const std::vector<groupT> &groups )
{
    std::vector<flatbuffers::Offset<IrqGroup>> records;

    for( const auto &group : groups )
    {
        std::vector<flatbuffers::Offset<IrqAffinity>> irqs;

        for( const auto &irq : group.m_irqs )
        {
            auto requested = builder.CreateString( irq.m_requested );
            auto effective = builder.CreateString( irq.m_effective );
            irqs.push_back( CreateIrqAffinity( builder, irq.m_number, requested, effective ) );
        }

        auto name     = builder.CreateString( group.m_name );
        auto expected = builder.CreateString( group.m_expected );
        auto status   = builder.CreateString( group.m_status );
        auto vectors  = builder.CreateVector( irqs );
        records.push_back(
            CreateIrqGroup( builder, name, expected, status, group.m_warning, group.m_checkTime, vectors ) );
    }

    auto vector = builder.CreateVector( records );
    builder.Finish( CreateTelem_irq_fb( builder, vector ) );
}

inline bool telem_irq::verify( flatlogs::bufferPtrT &logBuff, flatlogs::msgLenT len )
{
    flatbuffers::Verifier verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ), len );
    return VerifyTelem_irq_fbBuffer( verifier );
}

inline std::string telem_irq::msgString( void *msgBuffer, flatlogs::msgLenT len )
{
    static_cast<void>( len );
    std::string text = "[irq]";

    for( const auto *group : *GetTelem_irq_fb( msgBuffer )->groups() )
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

inline std::vector<telem_irq::groupT> telem_irq::groups( void *msgBuffer )
{
    std::vector<groupT> values;

    for( const auto *record : *GetTelem_irq_fb( msgBuffer )->groups() )
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

    for( const auto *group : *GetTelem_irq_fb( msgBuffer )->groups() )
    {
        values.push_back( group->name()->str() );
    }

    return values;
}

inline std::vector<std::string> telem_irq::statuses( void *msgBuffer )
{
    std::vector<std::string> values;

    for( const auto *group : *GetTelem_irq_fb( msgBuffer )->groups() )
    {
        values.push_back( group->status()->str() );
    }

    return values;
}

inline std::vector<std::string> telem_irq::expected( void *msgBuffer )
{
    std::vector<std::string> values;

    for( const auto *group : *GetTelem_irq_fb( msgBuffer )->groups() )
    {
        values.push_back( group->expected()->str() );
    }

    return values;
}

inline std::vector<double> telem_irq::checkTimes( void *msgBuffer )
{
    std::vector<double> values;

    for( const auto *group : *GetTelem_irq_fb( msgBuffer )->groups() )
    {
        values.push_back( group->check_time() );
    }

    return values;
}

inline bool telem_irq::warning( void *msgBuffer )
{
    for( const auto *group : *GetTelem_irq_fb( msgBuffer )->groups() )
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
