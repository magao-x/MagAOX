/** \file flatlogcodes_test.cpp
  * \brief Catch2 tests for the flatlogs format: log codes, CRC-32C, and the v1/v2 log header.
  *
  */
#include "../../tests/testXWC.hpp"

#include "../include/flatlogs/flatlogs.hpp"
#include "../include/flatlogs/logDefs.hpp"
#include "../include/flatlogs/logHeader.hpp"
#include "../include/flatlogs/logPriority.hpp"
#include "../include/flatlogs/logStdFormat.hpp"
#include "../include/flatlogs/timespecX.hpp"


#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace flatlogs_unit_test
{

using namespace flatlogs;

/// A minimal log type satisfying the createLog contract, with a std::string message.
struct dummyLog
{
   static const eventCodeT eventCode = 7;                          ///< The event code.
   static const logPrioT defaultLevel = logPrio::LOG_NOTICE;       ///< The default level.

   typedef std::string messageT;                                  ///< The message type.

   /// Report the serialized length of a message.
   static msgLenT length( const messageT &msg /**< [in] the message */ )
   {
      return msg.size();
   }

   /// Write a message into a buffer.
   static int format( void *msgBuffer,      /**< [out] the destination buffer */
                      const messageT &msg   /**< [in] the message */
                    )
   {
      memcpy( msgBuffer, msg.data(), msg.size() );
      return 0;
   }

   /// Get the message as text, for the formatters.
   static std::string msgString( void *msgBuffer, /**< [in] the message */
                                 msgLenT len      /**< [in] the message length */
                               )
   {
      return std::string( static_cast<char *>( msgBuffer ), len );
   }

   /// Get the message as a JSON string, for the JSON formatter.
   static std::string msgJSON( void *msgBuffer,                      /**< [in] the message */
                               msgLenT len,                          /**< [in] the message length */
                               const uint8_t *binarySchema,          /**< [in] unused */
                               const unsigned int binarySchemaLength /**< [in] unused */
                             )
   {
      static_cast<void>( binarySchema );
      static_cast<void>( binarySchemaLength );
      return "\"" + std::string( static_cast<char *>( msgBuffer ), len ) + "\"";
   }
};

/** Verify the CRC-32C implementation against canonical known-answer vectors.
  * These pin the algorithm so a later refactor cannot silently change the checksum
  * of every archive ever written.
  * \anchor logHeader_test_crc
  */
SCENARIO( "computing CRC-32C known-answer vectors", "[logHeader][crc]" )
{
   GIVEN( "the standard CRC-32C test strings" )
   {
      THEN( "the checksums match the published values" )
      {
         REQUIRE( crc32c( "123456789", 9 ) == 0xE3069283 ); //canonical CRC-32C check value
         REQUIRE( crc32c( "", 0 ) == 0x00000000 );
         REQUIRE( crc32c( "a", 1 ) == 0xC1D04330 );

         const char *fox = "The quick brown fox jumps over the lazy dog";
         REQUIRE( crc32c( fox, strlen( fox ) ) == 0x22620404 );
      }
   }
}

/** Verify that version-1 size arithmetic is byte-for-byte identical to the historical format.
  * This is the regression guard proving the templatization changes nothing for v1.
  */
SCENARIO( "version-1 size arithmetic is unchanged", "[logHeader][v1]" )
{
   GIVEN( "the version-1 header" )
   {
      THEN( "the fixed sizes match the historical values" )
      {
         REQUIRE( logHeaderV1::versionSize == 0 );
         REQUIRE( logHeaderV1::crcSize == 0 );
         REQUIRE( logHeaderV1::minHeadSize == 12 );
         REQUIRE( logHeaderV1::maxHeadSize == 20 );
      }

      THEN( "the default logHeader alias is the default version" )
      {
         REQUIRE( logHeader::formatVersion() == logVersionDefault );
         REQUIRE( logHeader::minHeadSize == logHeaderT<logVersionDefault>::minHeadSize );
      }

      THEN( "the length-field form changes at the boundaries the format block documents" )
      {
         //These pin the boundaries stated in the logHeader.hpp format description:
         //0-253 single byte, 254-65534 uint16 (marker 254), 65535 and up uint64 (marker 255).
         msgLenT L;
         L = 0;     REQUIRE( logHeaderV1::lenSize( L ) == 1 );
         L = 253;   REQUIRE( logHeaderV1::lenSize( L ) == 1 );
         L = 254;   REQUIRE( logHeaderV1::lenSize( L ) == 1 + sizeof( msgLen1T ) );
         L = 65534; REQUIRE( logHeaderV1::lenSize( L ) == 1 + sizeof( msgLen1T ) );
         L = 65535; REQUIRE( logHeaderV1::lenSize( L ) == 1 + sizeof( msgLen2T ) );
         L = 65536; REQUIRE( logHeaderV1::lenSize( L ) == 1 + sizeof( msgLen2T ) );
      }

      THEN( "totalSize matches the historical values across the length-field boundaries" )
      {
         msgLenT L;
         L = 0;     REQUIRE( logHeaderV1::totalSize( L ) == 0 + 12 );       //single length byte
         L = 253;   REQUIRE( logHeaderV1::totalSize( L ) == 253 + 12 );     //single length byte
         L = 254;   REQUIRE( logHeaderV1::totalSize( L ) == 254 + 14 );     //escapes to uint16 length
         L = 256;   REQUIRE( logHeaderV1::totalSize( L ) == 256 + 14 );     //matches logFileRaw_test invariant
         L = 65535; REQUIRE( logHeaderV1::totalSize( L ) == 65535 + 20 );   //escapes to uint64 length
      }
   }
}

/** Verify that version-2 adds exactly a one-byte version field and a four-byte CRC.
  */
SCENARIO( "version-2 adds exactly five bytes per entry", "[logHeader][v2]" )
{
   GIVEN( "the version-2 header" )
   {
      THEN( "the fixed sizes are the version-1 values plus the version byte" )
      {
         REQUIRE( logHeaderV2::versionSize == sizeof( logVersionT ) );
         REQUIRE( logHeaderV2::crcSize == sizeof( crcT ) );
         REQUIRE( logHeaderV2::minHeadSize == 13 );
         REQUIRE( logHeaderV2::maxHeadSize == 21 );
      }

      THEN( "totalSize is always the version-1 total plus five" )
      {
         for( msgLenT L : { (msgLenT)0, (msgLenT)253, (msgLenT)254, (msgLenT)256, (msgLenT)65535 } )
         {
            REQUIRE( logHeaderV2::totalSize( L ) == logHeaderV1::totalSize( L ) + 5 );
         }
      }
   }
}

/** Round-trip a version-2 entry: create it, verify it, and confirm every accessor returns the input.
  */
SCENARIO( "round-tripping a version-2 log entry", "[logHeader][v2]" )
{
   GIVEN( "a version-2 entry created by createLog" )
   {
      bufferPtrT b;
      timespecX ts( 1234567, 89 );
      std::string msg = "hello flatlogs v2";

      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV2::totalSize( b.get() );

      THEN( "verify accepts it and reports the correct entry size" )
      {
         size_t esz = 0;
         REQUIRE( logHeaderV2::verify( b.get(), total, esz ) );
         REQUIRE( esz == total );
      }

      THEN( "every field reads back what was written" )
      {
         REQUIRE( logHeaderV2::getVersion( b.get() ) == logVersion2 );
         REQUIRE( logHeaderV2::logLevel( b.get() ) == logPrio::LOG_NOTICE );
         REQUIRE( logHeaderV2::eventCode( b.get() ) == 7 );

         timespecX g = logHeaderV2::timespec( b.get() );
         REQUIRE( g.time_s == 1234567 );
         REQUIRE( g.time_ns == 89 );

         REQUIRE( logHeaderV2::msgLen( b.get() ) == msg.size() );

         void *mb = logHeaderV2::messageBuffer( b.get() );
         REQUIRE( memcmp( mb, msg.data(), msg.size() ) == 0 );
      }

      THEN( "the stored CRC equals the recomputed CRC" )
      {
         REQUIRE( logHeaderV2::getCRC( b.get() ) == logHeaderV2::computeCRC( b.get() ) );
      }
   }
}

/** A version-1 entry carries no version byte and no checksum: its first byte is the priority.
  */
SCENARIO( "a version-1 entry has no version byte or checksum", "[logHeader][v1]" )
{
   GIVEN( "a version-1 entry created by createLog" )
   {
      bufferPtrT b;
      timespecX ts( 42, 7 );
      std::string msg = "v1 entry";

      logHeaderV1::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      THEN( "totalSize has no trailing CRC" )
      {
         REQUIRE( logHeaderV1::totalSize( b.get() ) == logHeaderV1::headerSize( b.get() ) + msg.size() );
      }

      THEN( "the first byte is the priority, not a version field" )
      {
         REQUIRE( static_cast<logPrioT>( b.get()[0] ) == logPrio::LOG_NOTICE );
         REQUIRE( logHeaderV1::getVersion( b.get() ) == logVersion1 ); //implicit, not stored
      }
   }
}

/** Corruption detection: every single-bit flip in a version-2 entry must be caught by verify().
  */
SCENARIO( "version-2 verify detects single-bit corruption", "[logHeader][v2][crc]" )
{
   GIVEN( "a good version-2 entry" )
   {
      bufferPtrT b;
      timespecX ts( 999, 111 );
      std::string msg = "corrupt me please, byte by byte";
      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV2::totalSize( b.get() );

      THEN( "flipping any single bit in any byte is rejected" )
      {
         int missed = 0;
         for( size_t i = 0; i < total; ++i )
         {
            for( int bit = 0; bit < 8; ++bit )
            {
               char save = b.get()[i];
               b.get()[i] = save ^ ( 1 << bit );
               size_t esz = 0;
               if( logHeaderV2::verify( b.get(), total, esz ) ) ++missed;
               b.get()[i] = save;
            }
         }
         REQUIRE( missed == 0 );
      }
   }
}

/** Bounds safety: verify() must reject a truncated buffer without reading out of bounds.
  * Run under valgrind (the repo's existing setup) to confirm no invalid reads.
  */
SCENARIO( "version-2 verify is bounds-safe on truncated input", "[logHeader][v2]" )
{
   GIVEN( "a good version-2 entry" )
   {
      bufferPtrT b;
      timespecX ts( 5, 5 );
      std::string msg = "a message long enough to exercise the length field and beyond 254 bytes .........."
                        "..................................................................................."
                        "...................................................................................";
      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV2::totalSize( b.get() );

      THEN( "every prefix shorter than the full entry is rejected" )
      {
         for( size_t avail = 0; avail < total; ++avail )
         {
            size_t esz = 123; //sentinel: must be set to 0 on failure
            REQUIRE_FALSE( logHeaderV2::verify( b.get(), avail, esz ) );
            REQUIRE( esz == 0 );
         }
      }

      THEN( "the full-length buffer is accepted" )
      {
         size_t esz = 0;
         REQUIRE( logHeaderV2::verify( b.get(), total, esz ) );
         REQUIRE( esz == total );
      }
   }
}

/** A corrupt length field must be rejected without a wild read.
  */
SCENARIO( "version-2 verify rejects a corrupt length field", "[logHeader][v2]" )
{
   GIVEN( "a good version-2 entry with a short message" )
   {
      bufferPtrT b;
      timespecX ts( 1, 2 );
      std::string msg = "short";
      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV2::totalSize( b.get() );

      //Corrupt msgLen0 (just after the version byte) to the single-byte-length escape 0xFF,
      //which claims an enormous message the buffer cannot contain.
      std::vector<char> scratch( b.get(), b.get() + total );
      scratch[logHeaderV2::versionSize + sizeof( logPrioT ) + sizeof( eventCodeT ) + sizeof( secT ) + sizeof( nanosecT )] =
          static_cast<char>( 0xFF );

      THEN( "verify rejects it rather than reading out of bounds" )
      {
         size_t esz = 0;
         REQUIRE_FALSE( logHeaderV2::verify( scratch.data(), scratch.size(), esz ) );
         REQUIRE( esz == 0 );
      }
   }
}

/** Version mismatch: reading version-1 bytes with the version-2 parser must be rejected.
  * This is the safeguard against a buffer created under one version being read as another.
  */
SCENARIO( "version-2 verify rejects version-1 data", "[logHeader][v2]" )
{
   GIVEN( "a version-1 entry" )
   {
      bufferPtrT b;
      timespecX ts( 7, 8 );
      std::string msg = "i am a v1 entry";
      logHeaderV1::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t v1total = logHeaderV1::totalSize( b.get() );

      THEN( "the version-2 parser does not accept it" )
      {
         size_t esz = 0;
         //The first byte of a v1 entry is its priority, and every defined priority is 0-127, so it can never
         //equal a marker (>= 0xD0).  The marker check therefore rejects v1 data every time,
         //for any priority -- the checksum is not relied on here.
         REQUIRE_FALSE( logHeaderV2::verify( b.get(), v1total, esz ) );
      }
   }

   GIVEN( "version-1 entries written at every defined priority" )
   {
      const logPrioT prios[] = { logPrio::LOG_EMERGENCY, logPrio::LOG_ALERT,   logPrio::LOG_CRITICAL,
                                 logPrio::LOG_ERROR,     logPrio::LOG_WARNING, logPrio::LOG_NOTICE,
                                 logPrio::LOG_INFO,      logPrio::LOG_DEBUG,   logPrio::LOG_DEBUG2,
                                 logPrio::LOG_TELEM };

      THEN( "the version-2 parser rejects every one of them on the marker alone" )
      {
         for( logPrioT p : prios )
         {
            bufferPtrT b;
            timespecX ts( 7, 8 );
            std::string msg = "i am a v1 entry";
            logHeaderV1::createLog<dummyLog>( b, ts, msg, p );

            REQUIRE( logHeaderV1::logLevel( b.get() ) == p ); //priority really is the first byte

            size_t esz = 0;
            REQUIRE_FALSE( logHeaderV2::verify( b.get(), logHeaderV1::totalSize( b.get() ), esz ) );
            REQUIRE( esz == 0 );
         }
      }
   }
}

/** The hardware CRC-32C path must produce values identical to the portable table implementation,
  * across every tail length and starting alignment.  A mismatch would silently change the checksum
  * of every entry written on one build relative to another.
  */
SCENARIO( "the hardware and software CRC-32C paths agree", "[logHeader][crc]" )
{
   GIVEN( "pseudo-random data at every alignment and length" )
   {
      unsigned char buf[600];
      for( size_t i = 0; i < sizeof( buf ); ++i )
      {
         buf[i] = static_cast<unsigned char>( ( i * 1103515245u + 12345u ) >> 16 );
      }

      THEN( "crc32c and crc32cSoftware return the same value" )
      {
         size_t mismatches = 0;
         for( size_t off = 0; off < 8; ++off )
         {
            for( size_t len = 0; len <= 512; ++len )
            {
               if( crc32c( buf + off, len ) != crc32cSoftware( buf + off, len ) ) ++mismatches;
            }
         }
         REQUIRE( mismatches == 0 );
      }
   }
}

/** The on-disk version marker must be a value no version-1 priority can take, so that a version-2
  * reader rejects version-1 data on the marker alone rather than relying on the checksum.
  */
SCENARIO( "the version marker cannot collide with a version-1 priority", "[logHeader][v2]" )
{
   GIVEN( "the version-2 marker" )
   {
      THEN( "it is 0xD2 and lies outside the valid priority range" )
      {
         REQUIRE( logVersionMarker( logVersion2 ) == 0xD2 );
         //Design constraint: the marker must sit above the defined priority range so it cannot
         //collide with one.  The exhaustive check against each defined priority is below.
         REQUIRE( logVersionMarker( logVersion2 ) > 127 );
      }

      THEN( "out-of-range versions are rejected, not masked into range" )
      {
         //Masking would alias 18 onto 0xD2 and make it indistinguishable from version 2.
         REQUIRE( logVersionMarker( 18 ) == 0 );
         REQUIRE( logVersionMarker( 16 ) == 0 );
         REQUIRE( logVersionMarker( logVersion1 ) == 0 ); //v1 carries no marker
         REQUIRE( logVersionMarker( 0 ) == 0 );

         //Every in-range version maps to a distinct marker in 0xD2-0xDF.
         for( logVersionT v = logVersion2; v <= 0x0F; ++v )
         {
            REQUIRE( logVersionMarker( v ) == static_cast<logVersionT>( 0xD0 | v ) );
            REQUIRE( logVersionMarker( v ) != 0 );
         }
      }

      THEN( "getVersion rejects bytes in 0xD0-0xDF that are not valid markers" )
      {
         //0xD0 and 0xD1 have the right high nibble but are not markers: there is no version 0, and
         //version 1 stores no marker at all.  Decoding by high nibble alone would accept them.
         bufferPtrT b;
         timespecX ts( 1, 1 );
         std::string msg = "marker decode";
         logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

         REQUIRE( logHeaderV2::getVersion( b.get() ) == logVersion2 ); //genuine 0xD2

         b.get()[0] = static_cast<char>( 0xD1 );
         REQUIRE( logHeaderV2::getVersion( b.get() ) == 0 );

         b.get()[0] = static_cast<char>( 0xD0 );
         REQUIRE( logHeaderV2::getVersion( b.get() ) == 0 );

         b.get()[0] = static_cast<char>( 0xC2 ); //wrong high nibble
         REQUIRE( logHeaderV2::getVersion( b.get() ) == 0 );

         b.get()[0] = static_cast<char>( 0xDF ); //version 15, a valid marker
         REQUIRE( logHeaderV2::getVersion( b.get() ) == 15 );
      }

      THEN( "a zero byte never matches, even for a version with no valid marker" )
      {
         //logVersionMarker returns 0 for an out-of-range version, and 0 is a common byte in real
         //data: it is LOG_EMERGENCY, and it fills blank or truncated regions.  A direct comparison
         //would match those; isVersionMarker must not.
         REQUIRE_FALSE( isVersionMarker( 0, logVersion1 ) );
         REQUIRE_FALSE( isVersionMarker( 0, 0 ) );
         REQUIRE_FALSE( isVersionMarker( 0, 16 ) );
         REQUIRE_FALSE( isVersionMarker( 0, 18 ) );

         //A generic reader looping over every version must not match a zero byte for any of them.
         for( logVersionT v = 0; v <= 20; ++v )
         {
            REQUIRE_FALSE( isVersionMarker( 0, v ) );
         }

         //It still matches genuine markers.
         REQUIRE( isVersionMarker( 0xD2, logVersion2 ) );
         REQUIRE_FALSE( isVersionMarker( 0xD3, logVersion2 ) );
      }

      THEN( "a version-1 LOG_EMERGENCY entry, whose first byte is zero, is not taken for a marker" )
      {
         bufferPtrT b;
         timespecX ts( 1, 1 );
         std::string msg = "emergency";
         logHeaderV1::createLog<dummyLog>( b, ts, msg, logPrio::LOG_EMERGENCY );

         REQUIRE( static_cast<unsigned char>( b.get()[0] ) == 0 ); //the trap byte

         size_t esz = 0;
         REQUIRE_FALSE( logHeaderV2::verify( b.get(), logHeaderV1::totalSize( b.get() ), esz ) );
      }

      THEN( "no valid priority can be mistaken for it" )
      {
         const logPrioT prios[] = { logPrio::LOG_EMERGENCY, logPrio::LOG_ALERT,   logPrio::LOG_CRITICAL,
                                    logPrio::LOG_ERROR,     logPrio::LOG_WARNING, logPrio::LOG_NOTICE,
                                    logPrio::LOG_INFO,      logPrio::LOG_DEBUG,   logPrio::LOG_DEBUG2,
                                    logPrio::LOG_TELEM,     logPrio::LOG_DEFAULT, logPrio::LOG_UNKNOWN };
         for( logPrioT p : prios )
         {
            REQUIRE( static_cast<logVersionT>( p ) != logVersionMarker( logVersion2 ) );
         }
      }
   }

   GIVEN( "a version-2 entry" )
   {
      bufferPtrT b;
      timespecX ts( 3, 4 );
      std::string msg = "marker check";
      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      THEN( "the first byte on disk is the marker, and getVersion decodes the logical version" )
      {
         REQUIRE( static_cast<unsigned char>( b.get()[0] ) == 0xD2 );
         REQUIRE( logHeaderV2::getVersion( b.get() ) == logVersion2 );
      }
   }
}

/** Version-1 has no checksum, so verify() can only confirm that an entry is structurally sound
  * and fits within the readable bytes.  It must still never read out of bounds.
  */
SCENARIO( "version-1 verify checks structure and bounds", "[logHeader][v1]" )
{
   GIVEN( "a version-1 entry" )
   {
      bufferPtrT b;
      timespecX ts( 11, 12 );
      std::string msg = "a v1 entry for structural verification";
      logHeaderV1::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV1::totalSize( b.get() );

      THEN( "a complete entry is accepted and its size reported" )
      {
         size_t esz = 0;
         REQUIRE( logHeaderV1::verify( b.get(), total, esz ) );
         REQUIRE( esz == total );
      }

      THEN( "every truncated prefix is rejected without reading out of bounds" )
      {
         for( size_t avail = 0; avail < total; ++avail )
         {
            size_t esz = 123;
            REQUIRE_FALSE( logHeaderV1::verify( b.get(), avail, esz ) );
            REQUIRE( esz == 0 );
         }
      }
   }
}

/** The medium-length escape (msgLen0 == 254) changes the header size.  Corrupting a short entry's
  * length byte to that escape makes the header claim two more bytes than were written.
  */
SCENARIO( "version-2 verify rejects a damaged medium-length marker", "[logHeader][v2]" )
{
   GIVEN( "a short version-2 entry whose length byte is forced to the 254 escape" )
   {
      bufferPtrT b;
      timespecX ts( 21, 22 );
      std::string msg = "short entry";
      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV2::totalSize( b.get() );
      std::vector<char> scratch( b.get(), b.get() + total );

      //The msgLen0 field sits immediately after the fixed-size fields.
      const size_t lenPos = logHeaderV2::versionSize + sizeof( logPrioT ) + sizeof( eventCodeT ) +
                            sizeof( secT ) + sizeof( nanosecT );
      scratch[lenPos] = static_cast<char>( 254 );

      THEN( "verify rejects it" )
      {
         size_t esz = 0;
         REQUIRE_FALSE( logHeaderV2::verify( scratch.data(), scratch.size(), esz ) );
         REQUIRE( esz == 0 );
      }
   }
}

/** A length damaged to a *smaller* value still fits inside the buffer, so every bounds check passes.
  * Only the checksum can catch this, which is precisely the case the checksum exists for.
  */
SCENARIO( "version-2 verify rejects a shortened length that still fits", "[logHeader][v2][crc]" )
{
   GIVEN( "a version-2 entry whose length byte is reduced but still in range" )
   {
      bufferPtrT b;
      timespecX ts( 31, 32 );
      std::string msg = "a message of comfortable length";
      logHeaderV2::createLog<dummyLog>( b, ts, msg, logPrio::LOG_DEFAULT );

      size_t total = logHeaderV2::totalSize( b.get() );
      std::vector<char> scratch( b.get(), b.get() + total );

      const size_t lenPos = logHeaderV2::versionSize + sizeof( logPrioT ) + sizeof( eventCodeT ) +
                            sizeof( secT ) + sizeof( nanosecT );

      REQUIRE( static_cast<unsigned char>( scratch[lenPos] ) == msg.size() ); //single-byte length
      scratch[lenPos] = static_cast<char>( msg.size() - 4 );                  //still fits the buffer

      THEN( "the bounds checks pass but the checksum rejects it" )
      {
         size_t esz = 0;
         REQUIRE_FALSE( logHeaderV2::verify( scratch.data(), scratch.size(), esz ) );
         REQUIRE( esz == 0 );
      }
   }
}

/// Build a buffer of consecutive log entries of version \p v, one per message, recording where each starts.
template<logVersionT v>
std::vector<char> buildEntries( const std::vector<std::string> &msgs,   ///< [in] one message per entry
                                std::vector<size_t>            &offsets ///< [out] the offset of each entry
                              )
{
   std::vector<char> out;
   offsets.clear();
   for( size_t i = 0; i < msgs.size(); ++i )
   {
      bufferPtrT b;
      timespecX  ts( 1000 + i, 500 + i );
      logHeaderT<v>::template createLog<dummyLog>( b, ts, msgs[i], logPrio::LOG_DEFAULT );
      offsets.push_back( out.size() );
      size_t n = logHeaderT<v>::totalSize( b.get() );
      out.insert( out.end(), b.get(), b.get() + n );
   }
   return out;
}

/// Check that a parsed entry which did not fit has nothing set beyond what the checks reached.
void requireOverrunFieldsClear( const parsedEntry &pe /**< [in] a result whose status is overrun */ )
{
   REQUIRE( pe.status == entryStatus::overrun );
   REQUIRE( pe.entrySize == 0 );
   REQUIRE( pe.msg == nullptr );
   REQUIRE( pe.prio == 0 );
   REQUIRE( pe.eventCode == 0 );
   REQUIRE( pe.ts.time_s == 0 );
   REQUIRE( pe.ts.time_ns == 0 );
}

/** parse() decodes complete v1 and v2 entries at every length-field boundary: the fields, the sizes, the
  * message location and the status.
  */
SCENARIO( "parse decodes v1 and v2 entries at every length boundary", "[logEntry]" )
{
   for( size_t L : { (size_t)0, (size_t)253, (size_t)254, (size_t)65534, (size_t)65535 } )
   {
      std::string msg( L, 'q' );
      timespecX   ts( 1234567, 89 );

      bufferPtrT b1;
      logHeaderV1::createLog<dummyLog>( b1, ts, msg, logPrio::LOG_DEFAULT );
      parsedEntry p1 = logHeaderV1::parse( b1.get(), logHeaderV1::totalSize( b1.get() ) );
      REQUIRE( p1.status == entryStatus::noChecksum );
      REQUIRE( p1.version == logVersion1 );
      REQUIRE( p1.prio == logPrio::LOG_NOTICE );
      REQUIRE( p1.eventCode == 7 );
      REQUIRE( p1.ts.time_s == 1234567 );
      REQUIRE( p1.ts.time_ns == 89 );
      REQUIRE( p1.msgLen == L );
      REQUIRE( p1.headerSize == logHeaderV1::headerSize( b1.get() ) );
      REQUIRE( p1.entrySize == logHeaderV1::totalSize( b1.get() ) );
      REQUIRE( p1.msg == b1.get() + p1.headerSize );
      REQUIRE( memcmp( p1.msg, msg.data(), L ) == 0 );

      bufferPtrT b2;
      logHeaderV2::createLog<dummyLog>( b2, ts, msg, logPrio::LOG_DEFAULT );
      parsedEntry p2 = logHeaderV2::parse( b2.get(), logHeaderV2::totalSize( b2.get() ) );
      REQUIRE( p2.status == entryStatus::checksumOk );
      REQUIRE( p2.version == logVersion2 );
      REQUIRE( p2.prio == logPrio::LOG_NOTICE );
      REQUIRE( p2.eventCode == 7 );
      REQUIRE( p2.ts.time_s == 1234567 );
      REQUIRE( p2.msgLen == L );
      REQUIRE( p2.headerSize == p1.headerSize + 1 );
      REQUIRE( p2.entrySize == p1.entrySize + 5 );
      REQUIRE( p2.msg == b2.get() + p2.headerSize );
      REQUIRE( memcmp( p2.msg, msg.data(), L ) == 0 );
   }
}

/** The runtime-version size helpers agree with the fixed-version functions at every length boundary.
  */
SCENARIO( "runtime-version size helpers match each version's sizes", "[logEntry]" )
{
   REQUIRE( entryMinHeadSize( logVersion1 ) == static_cast<size_t>( logHeaderV1::minHeadSize ) );
   REQUIRE( entryMinHeadSize( logVersion2 ) == static_cast<size_t>( logHeaderV2::minHeadSize ) );
   REQUIRE( entryMinHeadSize( 9 ) == 0 );

   for( size_t L : { (size_t)0, (size_t)253, (size_t)254, (size_t)65534, (size_t)65535 } )
   {
      std::string msg( L, 'q' );
      timespecX   ts( 1234567, 89 );

      bufferPtrT b1;
      bufferPtrT b2;
      logHeaderV1::createLog<dummyLog>( b1, ts, msg, logPrio::LOG_DEFAULT );
      logHeaderV2::createLog<dummyLog>( b2, ts, msg, logPrio::LOG_DEFAULT );

      REQUIRE( entryHeaderSize( b1.get(), logVersion1 ) == logHeaderV1::headerSize( b1.get() ) );
      REQUIRE( entryTotalSize( b1.get(), logVersion1 ) == logHeaderV1::totalSize( b1.get() ) );
      REQUIRE( entryHeaderSize( b2.get(), logVersion2 ) == logHeaderV2::headerSize( b2.get() ) );
      REQUIRE( entryTotalSize( b2.get(), logVersion2 ) == logHeaderV2::totalSize( b2.get() ) );
      REQUIRE( entryHeaderSize( b1.get(), 9 ) == 0 );
      REQUIRE( entryTotalSize( b1.get(), 9 ) == 0 );
   }
}

/** Every truncated prefix of an entry gives overrun, never reading past the bytes given.  Each prefix is
  * copied into a buffer of exactly that size, so under AddressSanitizer any over-read is caught.
  */
SCENARIO( "parse reports overrun for every truncated entry", "[logEntry]" )
{
   for( size_t L : { (size_t)20, (size_t)300 } ) // single-byte and two-byte length fields
   {
      std::string msg( L, 'z' );
      timespecX   ts( 7, 8 );
      bufferPtrT  b1, b2;
      logHeaderV1::createLog<dummyLog>( b1, ts, msg, logPrio::LOG_DEFAULT );
      logHeaderV2::createLog<dummyLog>( b2, ts, msg, logPrio::LOG_DEFAULT );
      size_t t1 = logHeaderV1::totalSize( b1.get() ), h1 = logHeaderV1::headerSize( b1.get() );
      size_t t2 = logHeaderV2::totalSize( b2.get() ), h2 = logHeaderV2::headerSize( b2.get() );

      for( size_t avail = 0; avail < t1; ++avail )
      {
         std::vector<char> exact( b1.get(), b1.get() + avail );
         parsedEntry       pe = logHeaderV1::parse( avail ? exact.data() : nullptr, avail );
         requireOverrunFieldsClear( pe );
         REQUIRE( pe.headerSize == ( avail >= h1 ? h1 : 0 ) ); // set only once the full header fits
      }

      for( size_t avail = 0; avail < t2; ++avail )
      {
         std::vector<char> exact( b2.get(), b2.get() + avail );
         parsedEntry       pe = logHeaderV2::parse( avail ? exact.data() : nullptr, avail );
         requireOverrunFieldsClear( pe );
         REQUIRE( pe.headerSize == ( avail >= h2 ? h2 : 0 ) );
         REQUIRE( pe.msgLen == ( avail >= h2 ? L : 0 ) );
      }
   }

   GIVEN( "fewer bytes than the smallest header, starting with a byte that is not the v2 marker" )
   {
      std::vector<char> junk( logHeaderV2::minHeadSize - 1, 0 );
      THEN( "the result is overrun, not markerMismatch, since the marker may not even be present" )
      {
         for( size_t avail = 0; avail < junk.size(); ++avail )
         {
            std::vector<char> exact( junk.begin(), junk.begin() + avail );
            REQUIRE( logHeaderV2::parse( avail ? exact.data() : nullptr, avail ).status == entryStatus::overrun );
         }
      }
   }
}

/** v1 bytes read with v2 rules give markerMismatch, with nothing else set.
  */
SCENARIO( "parse reports a marker mismatch for v1 data read as v2", "[logEntry]" )
{
   bufferPtrT  b;
   std::string msg = "a v1 entry";
   logHeaderV1::createLog<dummyLog>( b, timespecX( 3, 4 ), msg, logPrio::LOG_DEFAULT );
   parsedEntry pe = logHeaderV2::parse( b.get(), logHeaderV1::totalSize( b.get() ) );

   REQUIRE( pe.status == entryStatus::markerMismatch );
   REQUIRE( pe.version == logVersion2 );
   REQUIRE( pe.headerSize == 0 );
   REQUIRE( pe.msgLen == 0 );
   REQUIRE( pe.entrySize == 0 );
   REQUIRE( pe.msg == nullptr );
   REQUIRE( pe.eventCode == 0 );
}

/** No single-bit flip anywhere in a v2 entry parses as checksumOk.
  */
SCENARIO( "no single-bit flip in a v2 entry parses as checksumOk", "[logEntry][crc]" )
{
   bufferPtrT  b;
   std::string msg = "flip every bit of this entry, one at a time";
   logHeaderV2::createLog<dummyLog>( b, timespecX( 5, 6 ), msg, logPrio::LOG_DEFAULT );
   size_t total = logHeaderV2::totalSize( b.get() );

   size_t accepted = 0;
   for( size_t i = 0; i < total; ++i )
   {
      for( int bit = 0; bit < 8; ++bit )
      {
         std::vector<char> copy( b.get(), b.get() + total );
         copy[i] ^= static_cast<char>( 1 << bit );
         if( logHeaderV2::parse( copy.data(), copy.size() ).status == entryStatus::checksumOk ) ++accepted;
      }
   }
   REQUIRE( accepted == 0 );
}

/** parseEntry() dispatches on a version known only at runtime, and rejects versions with no layout.
  */
SCENARIO( "parseEntry dispatches on the runtime version", "[logEntry]" )
{
   std::string msg = "runtime version";
   bufferPtrT  b1, b2;
   logHeaderV1::createLog<dummyLog>( b1, timespecX( 1, 1 ), msg, logPrio::LOG_DEFAULT );
   logHeaderV2::createLog<dummyLog>( b2, timespecX( 1, 1 ), msg, logPrio::LOG_DEFAULT );

   REQUIRE( parseEntry( b1.get(), logHeaderV1::totalSize( b1.get() ), logVersion1 ).status == entryStatus::noChecksum );
   REQUIRE( parseEntry( b2.get(), logHeaderV2::totalSize( b2.get() ), logVersion2 ).status == entryStatus::checksumOk );

   for( logVersionT v : { (logVersionT)0, (logVersionT)3, (logVersionT)16 } )
   {
      parsedEntry pe = parseEntry( b2.get(), logHeaderV2::totalSize( b2.get() ), v );
      REQUIRE( pe.status == entryStatus::unknownVersion );
      REQUIRE( pe.version == v );
      REQUIRE( pe.entrySize == 0 );
   }
}

/** findNextEntryV2() finds the next good entry after each kind of damage, and never returns a position at
  * or beyond the bytes it was given.
  */
SCENARIO( "findNextEntryV2 recovers from damage", "[logEntry]" )
{
   std::vector<std::string> msgs = { "first entry", "second entry, the one that gets damaged",
                                     std::string( 40, 'c' ), std::string( 300, 'd' ), "last entry" };
   std::vector<size_t>      off;
   const std::vector<char>  good = buildEntries<logVersion2>( msgs, off );

   // offset of entry 1's single-byte length field: version, priority, event code, seconds, nanoseconds
   const size_t lenField = logHeaderV2::versionSize + sizeof( logPrioT ) + sizeof( eventCodeT ) + sizeof( secT ) +
                           sizeof( nanosecT );

   auto findAfter = [&]( const std::vector<char> &buf, size_t at )
   {
      parsedEntry failed = logHeaderV2::parse( buf.data() + at, buf.size() - at );
      size_t      next   = findNextEntryV2( buf.data(), buf.size(), at, failed );
      REQUIRE( ( next == entryNotFound || next < buf.size() ) );
      return std::make_pair( failed.status, next );
   };

   GIVEN( "damage inside entry 1's message, with its length intact" )
   {
      std::vector<char> buf = good;
      buf[off[1] + logHeaderV2::headerSize( buf.data() + off[1] ) + 5] ^= 0x40;
      auto r = findAfter( buf, off[1] );
      THEN( "entry 1 fails its checksum and entry 2 is found where entry 1's length points" )
      {
         REQUIRE( r.first == entryStatus::checksumMismatch );
         REQUIRE( r.second == off[2] );
      }
   }

   GIVEN( "entry 1's length damaged to a smaller value, so its claimed end lies inside itself" )
   {
      std::vector<char> buf = good;
      buf[off[1] + lenField] = static_cast<char>( static_cast<unsigned char>( buf[off[1] + lenField] ) - 3 );
      auto r = findAfter( buf, off[1] );
      THEN( "the shortcut fails and scanning finds entry 2" )
      {
         REQUIRE( r.first == entryStatus::checksumMismatch );
         REQUIRE( r.second == off[2] );
      }
   }

   GIVEN( "entry 1's length damaged to a larger value, so its claimed end lies in the middle of entry 2" )
   {
      std::vector<char> buf = good;
      buf[off[1] + lenField] = static_cast<char>( static_cast<unsigned char>( buf[off[1] + lenField] ) + 5 );
      auto r = findAfter( buf, off[1] );
      THEN( "the shortcut lands mid-entry and fails, and scanning still finds entry 2" )
      {
         REQUIRE( r.first == entryStatus::checksumMismatch );
         REQUIRE( r.second == off[2] );
      }
   }

   GIVEN( "runs of 0x00 and 0xFF, typical of damaged storage, between two entries" )
   {
      for( char fill : { (char)0x00, (char)0xFF } )
      {
         std::vector<char> buf( good.begin(), good.begin() + off[1] ); // entry 0
         size_t            junkAt = buf.size();
         buf.insert( buf.end(), 100, fill );
         size_t nextAt = buf.size();
         buf.insert( buf.end(), good.begin() + off[1], good.begin() + off[2] ); // entry 1
         auto r = findAfter( buf, junkAt );
         REQUIRE( r.first == entryStatus::markerMismatch );
         REQUIRE( r.second == nextAt );
      }
   }

   GIVEN( "damage in the last entry, with the data ending exactly at its claimed end" )
   {
      std::vector<char> buf = good;
      buf[off[4] + logHeaderV2::headerSize( buf.data() + off[4] ) + 2] ^= 0x01;
      auto r = findAfter( buf, off[4] );
      THEN( "there is nothing after it to find" )
      {
         REQUIRE( r.first == entryStatus::checksumMismatch );
         REQUIRE( r.second == entryNotFound );
      }
   }

   GIVEN( "a starting position at or beyond the end, or no buffer" )
   {
      parsedEntry none;
      THEN( "nothing is found, and nothing is read" )
      {
         REQUIRE( findNextEntryV2( good.data(), good.size(), good.size(), none ) == entryNotFound );
         REQUIRE( findNextEntryV2( good.data(), good.size(), good.size() + 10, none ) == entryNotFound );
         REQUIRE( findNextEntryV2( nullptr, 100, 0, none ) == entryNotFound );
      }
   }
}

/** The worst natural pattern for the search: a run of 0xD2 bytes, where every position passes the marker
  * check and claims a message.  The real entry after the run is found exactly, and nothing inside the run
  * is accepted.
  */
SCENARIO( "findNextEntryV2 finds a real entry after a run of marker bytes", "[logEntry]" )
{
   std::vector<char> buf( 4096, static_cast<char>( 0xD2 ) );
   const size_t      runEnd = buf.size();

   std::vector<size_t> off;
   std::vector<char>   one = buildEntries<logVersion2>( { "after the run" }, off );
   buf.insert( buf.end(), one.begin(), one.end() );

   parsedEntry first = logHeaderV2::parse( buf.data(), buf.size() );
   REQUIRE( first.status != entryStatus::checksumOk );
   REQUIRE( findNextEntryV2( buf.data(), buf.size(), 0, first ) == runEnd );
}

/** Correctness over megabytes of random bytes with one real entry near the end: the search finds exactly
  * that entry, and nothing after it.  No time limit is asserted, since the suite also runs under valgrind.
  */
SCENARIO( "findNextEntryV2 finds a real entry after megabytes of random bytes", "[logEntry]" )
{
   const size_t      N = 10 * 1024 * 1024;
   std::vector<char> buf( N );
   uint32_t          x = 12345u; // deterministic pseudo-random bytes
   for( size_t i = 0; i < N; ++i )
   {
      x      = x * 1664525u + 1013904223u;
      buf[i] = static_cast<char>( x >> 24 );
   }

   std::vector<size_t> off;
   std::vector<char>   one     = buildEntries<logVersion2>( { "the only real entry" }, off );
   const size_t        plantAt = N - 1000;
   std::copy( one.begin(), one.end(), buf.begin() + plantAt );

   parsedEntry first = logHeaderV2::parse( buf.data(), N );
   REQUIRE( first.status != entryStatus::checksumOk );
   REQUIRE( findNextEntryV2( buf.data(), N, 0, first ) == plantAt );

   parsedEntry planted = logHeaderV2::parse( buf.data() + plantAt, N - plantAt );
   REQUIRE( planted.status == entryStatus::checksumOk );
   REQUIRE( findNextEntryV2( buf.data(), N, plantAt, planted ) == entryNotFound );
}

/** Printing a decoded entry gives the same text for v1 and v2, and the same as the raw-entry formatters, which
  * decode with the default version.
  */
SCENARIO( "formatting decoded v1 and v2 entries", "[logStdFormat]" )
{
   GIVEN( "the same entry written as v1 and as v2" )
   {
      timespecX ts( 1234567, 89 );
      std::string msg = "formatted the same";

      bufferPtrT b1;
      bufferPtrT b2;
      bufferPtrT bd; // written with the default version, as the writer does, for the raw-entry formatters
      logHeaderV1::createLog<dummyLog>( b1, ts, msg, logPrio::LOG_DEFAULT );
      logHeaderV2::createLog<dummyLog>( b2, ts, msg, logPrio::LOG_DEFAULT );
      logHeader::createLog<dummyLog>( bd, ts, msg, logPrio::LOG_DEFAULT );

      parsedEntry e1 = logHeaderV1::parse( b1.get(), logHeaderV1::totalSize( b1.get() ) );
      parsedEntry e2 = logHeaderV2::parse( b2.get(), logHeaderV2::totalSize( b2.get() ) );
      REQUIRE( e1.status == entryStatus::noChecksum );
      REQUIRE( e2.status == entryStatus::checksumOk );

      THEN( "each formatter gives the same text for v1, v2 and the raw-entry version" )
      {
         std::ostringstream s1, s2, sw;
         stdFormat<dummyLog>( s1, e1 );
         stdFormat<dummyLog>( s2, e2 );
         stdFormat<dummyLog>( sw, bd );
         REQUIRE( s1.str() == s2.str() );
         REQUIRE( s1.str() == sw.str() );
         REQUIRE( s1.str().find( "formatted the same" ) != std::string::npos );

         std::ostringstream m1, m2, mw;
         minFormat<dummyLog>( m1, e1 );
         minFormat<dummyLog>( m2, e2 );
         minFormat<dummyLog>( mw, bd );
         REQUIRE( m1.str() == m2.str() );
         REQUIRE( m1.str() == mw.str() );

         std::ostringstream h1, h2, hw;
         stdShortFormat<dummyLog>( h1, "someApp", e1 );
         stdShortFormat<dummyLog>( h2, "someApp", e2 );
         stdShortFormat<dummyLog>( hw, "someApp", bd );
         REQUIRE( h1.str() == h2.str() );
         REQUIRE( h1.str() == hw.str() );

         std::ostringstream j1, j2, jw;
         jsonFormat<dummyLog>( j1, e1, "dummyLog", nullptr, 0 );
         jsonFormat<dummyLog>( j2, e2, "dummyLog", nullptr, 0 );
         jsonFormat<dummyLog>( jw, bd, "dummyLog", nullptr, 0 );
         REQUIRE( j1.str() == j2.str() );
         REQUIRE( j1.str() == jw.str() );
         REQUIRE( j1.str().find( "\"msg\": \"formatted the same\"" ) != std::string::npos );
      }

      THEN( "an entry whose message couldn't be decoded prints a placeholder instead of reading it" )
      {
         parsedEntry bad = logHeaderV2::parse( b1.get(), logHeaderV1::totalSize( b1.get() ) );
         REQUIRE( bad.status == entryStatus::markerMismatch );
         REQUIRE( bad.msg == nullptr );

         std::ostringstream s, j;
         stdFormat<dummyLog>( s, bad );
         jsonFormat<dummyLog>( j, bad, "dummyLog", nullptr, 0 );
         REQUIRE( s.str().find( "<undecodable entry>" ) != std::string::npos );
         REQUIRE( j.str().find( "\"msg\": null" ) != std::string::npos );
      }
   }
}

} //namespace flatlogs_unit_test
