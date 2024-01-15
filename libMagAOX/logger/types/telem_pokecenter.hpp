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
    ///The event code
    static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_POKECENTER;

    ///The default level
    static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

    static timespec lastRecord; ///< The time of the last time this log was recorded.  Used by the telemetry system.

    ///The type of the input message
    struct messageT : public fbMessage
    {
        ///Construct from components
        messageT( const uint8_t & measuring,         ///<[in] whether or not measurements are in progress
                  const float & pupil_x,             ///<[in] the pupil x position
                  const float & pupil_y,             ///<[in] the pupil y position
                  const std::vector<float> & poke_x, ///<[in] the poke x positions, last one the average
                  const std::vector<float> & poke_y  ///<[in] the poke y positions, last one the average
                )
        {
            if(measuring == 0)
            {
                Telem_pokecenter_fbBuilder telem_pokecenter_builder(builder);
                telem_pokecenter_builder.add_measuring(measuring);
                auto fb = telem_pokecenter_builder.Finish();
                builder.Finish(fb);
                return;
            }

            auto _poke_xs = builder.CreateVector(poke_x);
            auto _poke_ys = builder.CreateVector(poke_y);

            auto fb = CreateTelem_pokecenter_fb(builder, measuring, pupil_x, pupil_y, _poke_xs, _poke_ys);

            builder.Finish(fb);
        }

        ///Construct from components with single vector for pokes
        messageT( const uint8_t & measuring,         ///<[in] whether or not measurements are in progress
                  const float & pupil_x,           ///<[in] the pupil x position
                  const float & pupil_y,           ///<[in] the pupil y position
                  const std::vector<float> & pokes ///<[in] the combined poke positions, last two the averages
               )
        {
            if(measuring == 0)
            {
                Telem_pokecenter_fbBuilder telem_pokecenter_builder(builder);
                telem_pokecenter_builder.add_measuring(measuring);
                auto fb = telem_pokecenter_builder.Finish();
                builder.Finish(fb);
                return;
            }

            std::vector<float> poke_x(pokes.size()/2);
            std::vector<float> poke_y(pokes.size()/2);

            for(size_t n = 0; n < poke_x.size(); ++n)
            {
                poke_x[n] = pokes[2*n + 0];
                poke_y[n] = pokes[2*n + 1];
            }

            auto _poke_xs = builder.CreateVector(poke_x);
            auto _poke_ys = builder.CreateVector(poke_y);

            auto fb = CreateTelem_pokecenter_fb(builder, measuring, pupil_x, pupil_y, _poke_xs, _poke_ys);

            builder.Finish(fb);
        }
    };
                 
   static bool verify( flatlogs::bufferPtrT & logBuff,  ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len            ///< [in] length of msgBuffer.
                     )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t*>(flatlogs::logHeader::messageBuffer(logBuff)), static_cast<size_t>(len));
      return VerifyTelem_pokecenter_fbBuffer(verifier);
   }

   ///Get the message formatted for human consumption.
   static std::string msgString( void * msgBuffer,  /**< [in] Buffer containing the flatbuffer serialized message.*/
                                 flatlogs::msgLenT len  /**< [in] [unused] length of msgBuffer.*/
                               )
   {
      static_cast<void>(len);

      auto fbs = GetTelem_pokecenter_fb(msgBuffer);

      std::string msg;
      if(fbs->measuring() == 0)
      {
          msg = "not measuring";
          return msg;
      }

      if(fbs->measuring() == 1)
      {
          msg = "single ";
      }
      else 
      {
          msg = "continuous ";
      }

      msg += "[pupil] ";
      
      msg += std::to_string(fbs->pupil_x());
      msg += " ";
      msg += std::to_string(fbs->pupil_y());

      // being very paranoid about existence and length here
      if( fbs->poke_x() && fbs->poke_y() )
      {
         if(fbs->poke_x()->size() == fbs->poke_y()->size())
         {
            size_t N = fbs->poke_x()->size();

            msg += " [poke-avg] ";
            msg += std::to_string(fbs->poke_x()->Get(N-1));
            msg += " ";
            msg += std::to_string(fbs->poke_y()->Get(N-1));

            msg += " [pokes]";
            for(size_t i=0; i< N-1; ++i)
            {
               msg += " ";
               msg += std::to_string(fbs->poke_x()->Get(i));
               msg += " ";
               msg += std::to_string(fbs->poke_y()->Get(i));
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
   
}; //telem_pokecenter



} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_pokecenter_hpp

