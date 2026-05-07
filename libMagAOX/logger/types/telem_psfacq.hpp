/** \file telem_psfacq.hpp
  * \brief The MagAO-X logger telem_psfacq log type.
  *
  * \ingroup logger_types_files
  *
  */
#ifndef logger_types_telem_psfacq_hpp
#define logger_types_telem_psfacq_hpp

#include "generated/telem_psfacq_generated.h"
#include "flatbuffer_log.hpp"

namespace MagAOX
{
namespace logger
{

/// Log entry recording psf acquisition per-star properties.
/** \ingroup logger_types
  */
struct telem_psfacq : public flatbuffer_log
{
   /// The event code
   static const flatlogs::eventCodeT eventCode = eventCodes::TELEM_PSFACQ;

   /// The default level
   static const flatlogs::logPrioT defaultLevel = flatlogs::logPrio::LOG_TELEM;

   static timespec lastRecord; ///< The timestamp of the last time this log was recorded. Used by telemetry.

   /// The type of the input message
   struct messageT : public fbMessage
   {
      /// Construct from components
      messageT( const float &x_pos, /**< [in] x position in pixels. */
                const float &y_pos, /**< [in] y position in pixels. */
                const float &m_pix, /**< [in] peak pixel value. */
                const float &fwhm,  /**< [in] full-width at half-maximum in pixels. */
                const float &seeing /**< [in] seeing in arcseconds. */ )
      {
         auto fp = CreateTelem_psfacq_fb( builder, x_pos, y_pos, m_pix, fwhm, seeing );
         builder.Finish( fp );
      }
   };

   static bool verify( flatlogs::bufferPtrT &logBuff, ///< [in] Buffer containing the flatbuffer serialized message.
                       flatlogs::msgLenT len          ///< [in] length of msgBuffer.
   )
   {
      auto verifier = flatbuffers::Verifier( static_cast<uint8_t *>( flatlogs::logHeader::messageBuffer( logBuff ) ),
                                             static_cast<size_t>( len ) );
      return VerifyTelem_psfacq_fbBuffer( verifier );
   }

   /// Get the message format for human consumption.
   static std::string msgString( void *msgBuffer,      /**< [in] Buffer containing the flatbuffer serialized message. */
                                 flatlogs::msgLenT len /**< [in] [unused] length of msgBuffer. */ )
   {
      static_cast<void>( len );

      auto fbs = GetTelem_psfacq_fb( msgBuffer );

      std::string msg = "[psfAcq] ";
      msg += "x_pos: " + std::to_string( fbs->x_pos() ) + " ";
      msg += "y_pos: " + std::to_string( fbs->y_pos() ) + " ";
      msg += "m_pix: " + std::to_string( fbs->m_pix() ) + " ";
      msg += "fwhm: " + std::to_string( fbs->fwhm() ) + " ";
      msg += "seeing: " + std::to_string( fbs->seeing() ) + " ";

      return msg;
   }

   static float x_pos( void *msgBuffer )
   {
      auto fbs = GetTelem_psfacq_fb( msgBuffer );
      return fbs->x_pos();
   }

   static float y_pos( void *msgBuffer )
   {
      auto fbs = GetTelem_psfacq_fb( msgBuffer );
      return fbs->y_pos();
   }

   static float m_pix( void *msgBuffer )
   {
      auto fbs = GetTelem_psfacq_fb( msgBuffer );
      return fbs->m_pix();
   }

   static float fwhm( void *msgBuffer )
   {
      auto fbs = GetTelem_psfacq_fb( msgBuffer );
      return fbs->fwhm();
   }

   static float seeing( void *msgBuffer )
   {
      auto fbs = GetTelem_psfacq_fb( msgBuffer );
      return fbs->seeing();
   }

   /// Get the logMetaDetail for a member by name
   /**
     * \returns a logMetaDetail filled in with the appropriate details
     * \returns an empty logMetaDetail if member not recognized
     */
   static logMetaDetail getAccessor( const std::string &member /**< [in] the name of the member */ )
   {
      if( member == "x_pos" )
      {
         return logMetaDetail( { "X POS", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &x_pos ) } );
      }
      else if( member == "y_pos" )
      {
         return logMetaDetail( { "Y POS", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &y_pos ) } );
      }
      else if( member == "m_pix" )
      {
         return logMetaDetail( { "M PIX", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &m_pix ) } );
      }
      else if( member == "fwhm" )
      {
         return logMetaDetail( { "FWHM", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &fwhm ) } );
      }
      else if( member == "seeing" )
      {
         return logMetaDetail( { "SEEING", logMeta::valTypes::Float, logMeta::metaTypes::Continuous, reinterpret_cast<void *>( &seeing ) } );
      }
      else
      {
         std::cerr << "No member " << member << " in telem_psfacq\n";
         return logMetaDetail();
      }
   }

}; //telem_psfacq

} //namespace logger
} //namespace MagAOX

#endif //logger_types_telem_psfacq_hpp
