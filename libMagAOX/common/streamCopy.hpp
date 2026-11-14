/** \file streamCopy.hpp
 * \brief Copy memory with non-temporal (streaming) stores.
 */

#ifndef common_streamCopy_hpp
#define common_streamCopy_hpp

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __SSE2__
    #include <immintrin.h>
#endif

namespace MagAOX
{

/// Copy memory with non-temporal (streaming) stores, which write whole lines without first taking them from caches
/** Other cores' cached copies of the destination are invalidated without the copying core waiting to take each line
 * back, and the destination is not left in this core's cache.  Readers then load the data from memory.  The
 * unaligned head and tail use regular stores, and a store fence at the end makes the data visible before anything the
 * caller publishes afterwards.  Uses memcpy where SSE2 is not available.
 */
inline void streamCopy( void       *dest,  /**< [out] start of the destination */
                        const void *src,   /**< [in] start of the source */
                        size_t      nbytes /**< [in] number of bytes to copy */
);

inline void streamCopy( void *dest, const void *src, size_t nbytes )
{
#ifdef __SSE2__
    char       *d = static_cast<char *>( dest );
    const char *s = static_cast<const char *>( src );

    // regular stores up to the first 16 byte boundary of the destination
    size_t head = std::min( ( 16 - reinterpret_cast<uintptr_t>( d ) % 16 ) % 16, nbytes );

    memcpy( d, s, head );
    d += head;
    s += head;
    nbytes -= head;

    for( ; nbytes >= 16; nbytes -= 16, d += 16, s += 16 )
    {
        _mm_stream_si128( reinterpret_cast<__m128i *>( d ), _mm_loadu_si128( reinterpret_cast<const __m128i *>( s ) ) );
    }

    memcpy( d, s, nbytes );

    _mm_sfence();
#else
    memcpy( dest, src, nbytes );
#endif
}

} // namespace MagAOX

#endif // common_streamCopy_hpp
