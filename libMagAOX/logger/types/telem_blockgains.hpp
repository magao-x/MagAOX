/** \file telem_blockgains.hpp
 * \brief The MagAO-X logger telem_blockgains log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 * History:
 * - 2022-11-27 created by JRM
 */
#ifndef logger_types_telem_blockgains_hpp
#define logger_types_telem_blockgains_hpp

#include "generated/telem_blockgains_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording electronics rack temperature
/** \ingroup logger_types
 */
struct telem_blockgains : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_BLOCKGAINS;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const std::vector<float>   &gains,          ///<[in] vector of gains
                  const std::vector<uint8_t> &gains_constant, ///<[in] vector of gains constant flags
                  const std::vector<float>   &mcs,            ///<[in] vector of mult. coeffs.
                  const std::vector<uint8_t> &mcs_constant,   ///<[in] vector of mult. coeff constant flags
                  const std::vector<float>   &lims,           ///<[in] vector of limits
                  const std::vector<uint8_t> &lims_constant   ///<[in] vector of limits constant flags
        )
        {
            auto _gains  = builder.CreateVector( gains );
            auto _gainsc = builder.CreateVector( gains_constant );
            auto _mcs    = builder.CreateVector( mcs );
            auto _mcsc   = builder.CreateVector( mcs_constant );
            auto _lims   = builder.CreateVector( lims );
            auto _limsc  = builder.CreateVector( lims_constant );

            auto fb = CreateTelem_blockgains_fb( builder, _gains, _gainsc, _mcs, _mcsc, _lims, _limsc );

            builder.Finish( fb );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_blockgains_fbBuffer( verifier );
    }

    /// Get the message formatte for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        std::string msg = "[gains] ";

        // being very paranoid about existence and length here
        if( fbs->gains() && fbs->gains_constant() )
        {
            if( fbs->gains()->size() == fbs->gains_constant()->size() )
            {
                for( size_t i = 0; i < fbs->gains()->size(); ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->gains()->Get( i ) );
                    msg += " (";
                    msg += std::to_string( fbs->gains_constant()->Get( i ) );
                    msg += ")";
                }
            }
            else
            {
                for( size_t i = 0; i < fbs->gains()->size(); ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->gains()->Get( i ) );
                    msg += " (?)";
                }
            }
        }
        else if( fbs->gains() )
        {
            for( size_t i = 0; i < fbs->gains()->size(); ++i )
            {
                msg += " ";
                msg += std::to_string( fbs->gains()->Get( i ) );
                msg += " (?)";
            }
        }

        msg += " [mcs] ";
        if( fbs->mcs() && fbs->mcs_constant() )
        {
            if( fbs->mcs()->size() == fbs->mcs_constant()->size() )
            {
                for( size_t i = 0; i < fbs->mcs()->size(); ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->mcs()->Get( i ) );
                    msg += " (";
                    msg += std::to_string( fbs->mcs_constant()->Get( i ) );
                    msg += ")";
                }
            }
            else
            {
                for( size_t i = 0; i < fbs->mcs()->size(); ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->mcs()->Get( i ) );
                    msg += " (?)";
                }
            }
        }
        else if( fbs->mcs() )
        {
            for( size_t i = 0; i < fbs->mcs()->size(); ++i )
            {
                msg += " ";
                msg += std::to_string( fbs->mcs()->Get( i ) );
                msg += " (?)";
            }
        }

        msg += " [lims] ";

        if( fbs->lims() && fbs->lims_constant() )
        {
            if( fbs->lims()->size() == fbs->lims_constant()->size() )
            {
                for( size_t i = 0; i < fbs->lims()->size(); ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->lims()->Get( i ) );
                    msg += " (";
                    msg += std::to_string( fbs->lims_constant()->Get( i ) );
                    msg += ")";
                }
            }
            else
            {
                for( size_t i = 0; i < fbs->lims()->size(); ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->lims()->Get( i ) );
                    msg += " (?)";
                }
            }
        }
        else if( fbs->lims() )
        {
            for( size_t i = 0; i < fbs->lims()->size(); ++i )
            {
                msg += " ";
                msg += std::to_string( fbs->lims()->Get( i ) );
                msg += " (?)";
            }
        }

        return msg;
    }

    static std::vector<float> gains( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<float> p;

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        if( fbs->gains() != nullptr )
        {
            for( auto it = fbs->gains()->begin(); it != fbs->gains()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    static std::vector<bool>
    gains_constant( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<bool> p;

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        if( fbs->gains_constant() != nullptr )
        {
            for( auto it = fbs->gains_constant()->begin(); it != fbs->gains_constant()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    static std::vector<float> mcs( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<float> p;

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        if( fbs->mcs() != nullptr )
        {
            for( auto it = fbs->mcs()->begin(); it != fbs->mcs()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    static std::vector<bool>
    mcs_constant( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<bool> p;

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        if( fbs->mcs_constant() != nullptr )
        {
            for( auto it = fbs->mcs_constant()->begin(); it != fbs->mcs_constant()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    static std::vector<float> lims( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<float> p;

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        if( fbs->lims() != nullptr )
        {
            for( auto it = fbs->lims()->begin(); it != fbs->lims()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    static std::vector<bool>
    lims_constant( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<bool> p;

        auto fbs = GetTelem_blockgains_fb( msgBuffer );

        if( fbs->lims_constant() != nullptr )
        {
            for( auto it = fbs->lims_constant()->begin(); it != fbs->lims_constant()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    /// Get the logMetaDetail for a member by name
    /**
     * \returns the a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
    static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
    {
        if( member == "gains" )
            return logMetaDetail( { "BLOCK GAINS",
                                    "mode block gains",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &gains ),
                                    true } );
        else if( member == "gains_constant" )
            return logMetaDetail( { "BLOCK GAINS CONSTANT",
                                    "whether or not all modes have same gain in block",
                                    logMeta::valTypes::Vector_Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &gains_constant ),
                                    true } );
        else if( member == "mcs" )
            return logMetaDetail( { "BLOCK MULT COEFS",
                                    "mode block mult. coefs.",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &mcs ),
                                    true } );

        else if( member == "mcs_constant" )
            return logMetaDetail( { "BLOCK MULT COEFS CONSTANT",
                                    "whether or not all modes have same mult. coefs. in block",
                                    logMeta::valTypes::Vector_Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &mcs_constant ),
                                    true } );

        else if( member == "lims" )
            return logMetaDetail( { "BLOCK LIMITS",
                                    "mode block limits",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &lims ),
                                    true } );

        else if( member == "lims_constant" )
            return logMetaDetail( { "BLOCK LIMITS CONSTANT",
                                    "whether or not all modes have same limit in block",
                                    logMeta::valTypes::Vector_Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &lims_constant ),
                                    true } );
        else
        {
            std::cerr << "No member " << member << " in telem_blockgains\n";
            return logMetaDetail();
        }
    }

}; // telem_blockgains

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_blockgains_hpp
