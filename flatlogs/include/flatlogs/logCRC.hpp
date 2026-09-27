/** \file logCRC.hpp
  * \brief CRC-32C (Castagnoli) checksum for the flatlogs format.
  *
  * \ingroup flatlogs_files
  *
  * Self-contained CRC-32C implementation for the version-2 flatlogs record checksum.
  *
  * \par Algorithm parameters
  * - width           : 32 bits
  * - polynomial      : 0x1EDC6F41 (with reflected form  0x82F63B78)
  * - initial value   : 0xFFFFFFFF
  * - input reflected : true
  * - output reflected: true
  * - final XOR       : 0xFFFFFFFF
  * - check value     : crc32c("123456789") == 0xE3069283
  *
  * \par Implementations
  * \ref crc32cSoftware is a portable table-driven implementation, always available.
  * \ref crc32c uses the CPU's CRC-32C instruction where the build enables one (ARMv8 \c +crc,
  * x86 SSE4.2) and falls back to the table otherwise. Both produce identical values, as asserted by
  * unit tests.
  */
#ifndef flatlogs_logCRC_hpp
#define flatlogs_logCRC_hpp

#include <cstddef>
#include <cstring>
#include "logDefs.hpp"

// The optimized hardware path implementation is byte-order sensitive and only matches the target
// algorithm on little-endian. Otherwise use table implementation which is unaffected by endianness.
#if( defined( __BYTE_ORDER__ ) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__ )
   #if defined( __ARM_FEATURE_CRC32 )
      #include <arm_acle.h>
      #define FLATLOGS_CRC32C_HW_ARM 1
   #elif defined( __SSE4_2__ )
      #include <nmmintrin.h>
      #define FLATLOGS_CRC32C_HW_X86 1
   #endif
#endif

namespace flatlogs
{

/// The reflected CRC-32C generator polynomial.
constexpr crcT crc32c_reflectedPoly = 0x82F63B78;

/// A 256-entry CRC-32C lookup table, generated at compile time.
/** The array is wrapped in a struct so a constexpr constructor can fill it, which lets the compiler
  * build the table at build time rather than at startup.
  */
struct crc32cTableT
{
   crcT data[256]; ///< The per-byte lookup values.

   /// Construct the table by folding each byte value through the polynomial.
   constexpr crc32cTableT() : data{}
   {
      for( crcT n = 0; n < 256; ++n )
      {
         crcT c = n;
         for( int k = 0; k < 8; ++k )
         {
            c = ( c & 1 ) ? ( crc32c_reflectedPoly ^ ( c >> 1 ) ) : ( c >> 1 );
         }
         data[n] = c;
      }
   }
};

constexpr crc32cTableT crc32cTable{};

/// Load a value of type T from a possibly-unaligned address.
/**
  * \returns the value of type T stored at \p bytes.
  */
template<typename T>
inline T loadUnaligned( const unsigned char *bytes /**< [in] the address to load from */ )
{
   T v;
   std::memcpy( &v, bytes, sizeof( T ) );
   return v;
}

/// Compute the CRC-32C checksum of a byte range, using the table implementation.
/**
  * \returns the 32-bit CRC-32C checksum of the \p len bytes at \p data.
  *
  * \ingroup logbuff
  */
inline crcT crc32cSoftware( const void *data, ///< [in] pointer to the first byte to checksum.
                            size_t len         ///< [in] the number of bytes to checksum.
                          )
{
   const unsigned char *bytes = static_cast<const unsigned char *>( data );

   crcT crc = 0xFFFFFFFF;

   for( size_t i = 0; i < len; ++i )
   {
      crc = ( crc >> 8 ) ^ crc32cTable.data[( crc ^ bytes[i] ) & 0xFF];
   }

   return crc ^ 0xFFFFFFFF;
}

/// Compute the CRC-32C checksum of a byte range.
/** Uses the CPU's CRC-32C instruction when the build enables one, otherwise \ref crc32cSoftware.
  * The result is identical either way.
  *
  * \returns the 32-bit CRC-32C checksum of the \p len bytes at \p data.
  *
  * \ingroup logbuff
  */
inline crcT crc32c( const void *data, ///< [in] pointer to the first byte to checksum.
                    size_t len         ///< [in] the number of bytes to checksum.
                  )
{
#if defined( FLATLOGS_CRC32C_HW_ARM ) || defined( FLATLOGS_CRC32C_HW_X86 )

   const unsigned char *bytes = static_cast<const unsigned char *>( data );

   crcT crc = 0xFFFFFFFF;

   size_t n = len;

   // The instruction comes in 64, 32, 16 and 8 bit widths. Consume as many 8-byte blocks as
   // possible, then at most one 4, one 2 and one 1.
   while( n >= 8 )
   {
      uint64_t v = loadUnaligned<uint64_t>( bytes );
   #if defined( FLATLOGS_CRC32C_HW_ARM )
      crc = __crc32cd( crc, v );
   #else
      crc = static_cast<crcT>( _mm_crc32_u64( crc, v ) );
   #endif
      bytes += 8;
      n -= 8;
   }

   if( n >= 4 )
   {
      uint32_t v = loadUnaligned<uint32_t>( bytes );
   #if defined( FLATLOGS_CRC32C_HW_ARM )
      crc = __crc32cw( crc, v );
   #else
      crc = _mm_crc32_u32( crc, v );
   #endif
      bytes += 4;
      n -= 4;
   }

   if( n >= 2 )
   {
      uint16_t v = loadUnaligned<uint16_t>( bytes );
   #if defined( FLATLOGS_CRC32C_HW_ARM )
      crc = __crc32ch( crc, v );
   #else
      crc = _mm_crc32_u16( crc, v );
   #endif
      bytes += 2;
      n -= 2;
   }

   if( n >= 1 )
   {
   #if defined( FLATLOGS_CRC32C_HW_ARM )
      crc = __crc32cb( crc, *bytes );
   #else
      crc = _mm_crc32_u8( crc, *bytes );
   #endif
   }

   return crc ^ 0xFFFFFFFF;

#else

   return crc32cSoftware( data, len );

#endif
}

} //namespace flatlogs

#endif //flatlogs_logCRC_hpp
