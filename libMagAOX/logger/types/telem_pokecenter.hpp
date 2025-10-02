/** \file telem_pokecenter.hpp
 * \brief The MagAO-X logger telem_pokecenter log type.
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup logger_types_files
 *
 */
#ifndef logger_types_telem_pokecenter_hpp
#define logger_types_telem_pokecenter_hpp

#include "generated/telem_pokecenter_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording DM poke centering results
/** \ingroup logger_types
 */
struct telem_pokecenter : public flatbuffer_log
{
    /// The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_POKECENTER;

    /// The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    /// The type of the input message
    struct messageT : public fbMessage
    {
        /// Construct from components
        messageT( const uint8_t            &measuring, ///<[in] whether or not measurements are in progress
                  const float              &pupil_x,   ///<[in] the pupil x position
                  const float              &pupil_y,   ///<[in] the pupil y position
                  const std::vector<float> &poke_x,    ///<[in] the poke x positions, last one the average
                  const std::vector<float> &poke_y     ///<[in] the poke y positions, last one the average
        )
        {
            if( measuring == 0 )
            {
                Telem_pokecenter_fbBuilder telem_pokecenter_builder( builder );
                telem_pokecenter_builder.add_measuring( measuring );
                auto fb = telem_pokecenter_builder.Finish();
                builder.Finish( fb );
                return;
            }

            auto _poke_xs = builder.CreateVector( poke_x );
            auto _poke_ys = builder.CreateVector( poke_y );

            auto fb = CreateTelem_pokecenter_fb( builder, measuring, pupil_x, pupil_y, _poke_xs, _poke_ys );

            builder.Finish( fb );
        }

        /// Construct from components with single vector for pokes
        messageT( const uint8_t            &measuring, ///<[in] whether or not measurements are in progress
                  const float              &pupil_x,   ///<[in] the pupil x position
                  const float              &pupil_y,   ///<[in] the pupil y position
                  const std::vector<float> &pokes      ///<[in] the combined poke positions, last two the averages
        )
        {
            if( measuring == 0 )
            {
                Telem_pokecenter_fbBuilder telem_pokecenter_builder( builder );
                telem_pokecenter_builder.add_measuring( measuring );
                auto fb = telem_pokecenter_builder.Finish();
                builder.Finish( fb );
                return;
            }

            std::vector<float> poke_x( pokes.size() / 2 );
            std::vector<float> poke_y( pokes.size() / 2 );

            for( size_t n = 0; n < poke_x.size(); ++n )
            {
                poke_x[n] = pokes[2 * n + 0];
                poke_y[n] = pokes[2 * n + 1];
            }

            auto _poke_xs = builder.CreateVector( poke_x );
            auto _poke_ys = builder.CreateVector( poke_y );

            auto fb = CreateTelem_pokecenter_fb( builder, measuring, pupil_x, pupil_y, _poke_xs, _poke_ys );

            builder.Finish( fb );
        }
    };

    static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                        flatlogs::msgLenT     len      ///< [in] length of msgBuffer.
    )
    {
        auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                               static_cast<size_t>( len ) );
        return VerifyTelem_pokecenter_fbBuffer( verifier );
    }

    /// Get the message formatted for human consumption.
    static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message.*/
                                  flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer.*/
    )
    {
        static_cast<void>( len );

        auto fbs = GetTelem_pokecenter_fb( msgBuffer );

        std::string msg;
        if( fbs->measuring() == 0 )
        {
            msg = "not measuring";
            return msg;
        }

        if( fbs->measuring() == 1 )
        {
            msg = "single ";
        }
        else
        {
            msg = "continuous ";
        }

        msg += "[pupil] ";

        msg += std::to_string( fbs->pupil_x() );
        msg += " ";
        msg += std::to_string( fbs->pupil_y() );

        // being very paranoid about existence and length here
        if( fbs->poke_x() && fbs->poke_y() )
        {
            if( fbs->poke_x()->size() == fbs->poke_y()->size() )
            {
                size_t N = fbs->poke_x()->size();

                msg += " [poke-avg] ";
                msg += std::to_string( fbs->poke_x()->Get( N - 1 ) );
                msg += " ";
                msg += std::to_string( fbs->poke_y()->Get( N - 1 ) );

                msg += " [pokes]";
                for( size_t i = 0; i < N - 1; ++i )
                {
                    msg += " ";
                    msg += std::to_string( fbs->poke_x()->Get( i ) );
                    msg += " ";
                    msg += std::to_string( fbs->poke_y()->Get( i ) );
                }
            }
            else
            {
                msg += " [poke-avg] ? [pokes] ?";
            }
        }
        else
        {
            msg += " [poke-avg] ? [pokes] ?";
        }

        return msg;
    }

    static bool measuring( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokecenter_fb( msgBuffer );
        return fbs->measuring();
    }

    static float pupil_x( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokecenter_fb( msgBuffer );
        return fbs->pupil_x();
    }

    static float pupil_y( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        auto fbs = GetTelem_pokecenter_fb( msgBuffer );
        return fbs->pupil_y();
    }

    static std::vector<float> poke_x( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<float> p;

        auto fbs = GetTelem_pokecenter_fb( msgBuffer );

        if( fbs->poke_x() != nullptr )
        {
            for( auto it = fbs->poke_x()->begin(); it != fbs->poke_x()->end(); ++it )
            {
                p.push_back( *it );
            }
        }
        return p;
    }

    static std::vector<float> poke_y( void *msgBuffer /**< [in] Buffer containing the flatbuffer serialized message.*/ )
    {
        std::vector<float> p;

        auto fbs = GetTelem_pokecenter_fb( msgBuffer );

        if( fbs->poke_y() != nullptr )
        {
            for( auto it = fbs->poke_y()->begin(); it != fbs->poke_y()->end(); ++it )
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
        if( member == "measuring" )
            return logMetaDetail( { "MEASURING",
                                    "measuring state",
                                    logMeta::valTypes::Bool,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &measuring ),
                                    false } );
        else if( member == "pupil_x" )
            return logMetaDetail( { "PUPIL X",
                                    "measured pupil x",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &pupil_x ),
                                    false } );
        else if( member == "pupil_y" )
            return logMetaDetail( { "PUPIL Y",
                                    "measured pupil y",
                                    logMeta::valTypes::Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &pupil_y ),
                                    false } );
        else if( member == "poke_x" )
            return logMetaDetail( { "POKE X",
                                    "actuator poke x, last is avg",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &poke_x ),
                                    false } );
        else if( member == "poke_y" )
            return logMetaDetail( { "POKE Y",
                                    "actuator poke y, last is avg",
                                    logMeta::valTypes::Vector_Float,
                                    logMeta::metaTypes::State,
                                    reinterpret_cast<void *>( &poke_y ),
                                    false } );
        else
        {
            std::cerr << "No member " << member << " in telem_pokecenter\n";
            return logMetaDetail();
        }
    }

}; // telem_pokecenter

} // namespace logger
} // namespace MagAOX

#endif // logger_types_telem_pokecenter_hpp
