/** \file ocamDescrambler.hpp
 * \brief Prepared single-core OCAM pixel mappings derived from the vendor SDK.
 *
 * \ingroup ocam2KCtrl_files
 */

#ifndef ocamDescrambler_hpp
#define ocamDescrambler_hpp

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include "fli/ocam2_sdk.h"

namespace MagAOX
{
namespace app
{

/// Prepare and apply an immutable OCAM source-index mapping outside SDK global state.
/** Preparation requires a valid, initialized SDK handle and descrambling table. The SDK remains responsible for
 * reading that table; this class cannot validate malformed table accesses performed inside the SDK itself.
 * Preparation and clearing must not overlap pixel conversion. Source and destination buffers must not overlap.
 *
 * \ingroup ocam2KCtrl
 */
class ocamDescrambler
{
  private:
    /// Source index for each output pixel, owning a compact copy of the configured SDK mapping.
    std::vector<uint16_t> m_map;

    /// Number of 16-bit samples required in each raw frame for the prepared mode.
    size_t m_rawPixels{ 0 };

    /// First source index of each 30-pixel amplifier run in the bin2 output rows.
    std::array<std::array<uint16_t, 4>, OCAM2_IMAGE_HEIGHT_BINNING2x2> m_rowBases{};

    /// Validated run strides, retained as data to avoid costly compiler expansion of fixed-stride gathers.
    std::array<int, 4> m_runStrides{};

    /// True only after every bin2 run has been verified to have its expected signed stride.
    bool m_structured{ false };

  public:
    /// Derive a source-index map by descrambling synthetic source indices with the existing SDK.
    /** \returns 0 on success, or -1 with no prepared map for an unsupported mode, invalid handle/index, or allocation
     * failure. SDK handle lifetime and table validity are the caller's responsibility.
     */
    int prepare( ocam2_id id /**< [in] Initialized SDK camera handle. */ );

    /// Invalidate the prepared map while retaining its allocated capacity for later configuration.
    void clear();

    /// Report whether a complete source-index map is available.
    bool ready() const;

    /// Report whether the prepared bin2 map supports the validated fixed-stride kernel.
    bool structured() const;

    /// Return the required number of raw 16-bit samples, or zero when unprepared.
    size_t rawPixels() const;

    /// Return the number of output pixels, or zero when unprepared.
    size_t imagePixels() const;

    /// Copy a frame through the compact map without changing its 16-bit pixel representations.
    /** Requires a prepared map and suitably sized, nonoverlapping buffers. This does not extract a frame number.
     */
    void mapped( short       *dest /**< [out] Buffer with at least imagePixels() samples. */,
                 const short *raw /**< [in] Buffer with at least rawPixels() samples. */ ) const;

    /// Convert a frame with the validated structured kernel, or the compact map when structure is unavailable.
    /** Requires a prepared map and suitably sized, nonoverlapping buffers. This does not extract a frame number.
     */
    void descramble( short       *dest /**< [out] Buffer with at least imagePixels() samples. */,
                     const short *raw /**< [in] Buffer with at least rawPixels() samples. */ ) const;
};

inline int ocamDescrambler::prepare( ocam2_id id )
{
    static_assert( sizeof( short ) == sizeof( uint16_t ), "OCAM samples must be 16 bits" );
    clear();

    if( id < 0 || id >= OCAM2_SDK_MAX_CAMERA )
    {
        return -1;
    }

    const ocam2_mode mode        = ocam2_getMode( id );
    size_t           imagePixels = 0;

    switch( mode )
    {
    case OCAM2_NORMAL:
        m_rawPixels = OCAM2_PIXELS_RAW_NORMAL;
        imagePixels = OCAM2_PIXELS_IMAGE_NORMAL;
        break;
    case OCAM2_BINNING2x2:
        m_rawPixels = OCAM2_PIXELS_RAW_BINNING2x2;
        imagePixels = OCAM2_PIXELS_IMAGE_BINNING2x2;
        break;
    case OCAM2_BINNING1x3:
        m_rawPixels = OCAM2_PIXELS_RAW_BINNING1x3;
        imagePixels = OCAM2_PIXELS_IMAGE_BINNING1x3;
        break;
    case OCAM2_BINNING1x4:
        m_rawPixels = OCAM2_PIXELS_RAW_BINNING1x4;
        imagePixels = OCAM2_PIXELS_IMAGE_BINNING1x4;
        break;
    default:
        return -1;
    }

    if( m_rawPixels == 0 || m_rawPixels - 1 > std::numeric_limits<uint16_t>::max() )
    {
        clear();
        return -1;
    }

    try
    {
        std::vector<short> raw( m_rawPixels );
        std::vector<short> image( imagePixels );
        m_map.resize( imagePixels );

        for( size_t i = 0; i < raw.size(); ++i )
        {
            const uint16_t index = static_cast<uint16_t>( i );
            std::memcpy( &raw[i], &index, sizeof( index ) );
        }

        unsigned frameNumber = 0;
        ocam2_descramble( id, &frameNumber, image.data(), raw.data() );
        std::memcpy( m_map.data(), image.data(), image.size() * sizeof( short ) );
    }
    catch( const std::bad_alloc & )
    {
        clear();
        return -1;
    }

    for( const uint16_t index : m_map )
    {
        if( index >= m_rawPixels )
        {
            clear();
            return -1;
        }
    }

    m_structured = mode == OCAM2_BINNING2x2;

    for( size_t row = 0; row < m_rowBases.size() && m_structured; ++row )
    {
        for( size_t amplifier = 0; amplifier < 4 && m_structured; ++amplifier )
        {
            const size_t offset        = row * OCAM2_IMAGE_WIDTH_BINNING2x2 + amplifier * 30;
            const int    stride        = amplifier % 2 == 0 ? -16 : 16;
            const int    base          = m_map[offset];
            m_rowBases[row][amplifier] = base;

            if( row == 0 )
            {
                m_runStrides[amplifier] = int( m_map[offset + 1] ) - base;
            }

            for( int pixel = 1; pixel < 30; ++pixel )
            {
                if( m_map[offset + pixel] != base + stride * pixel )
                {
                    m_structured = false;
                    break;
                }
            }
        }
    }

    return 0;
}

inline void ocamDescrambler::clear()
{
    m_map.clear();
    m_rawPixels  = 0;
    m_structured = false;
}

inline bool ocamDescrambler::ready() const
{
    return !m_map.empty();
}

inline bool ocamDescrambler::structured() const
{
    return m_structured;
}

inline size_t ocamDescrambler::rawPixels() const
{
    return m_rawPixels;
}

inline size_t ocamDescrambler::imagePixels() const
{
    return m_map.size();
}

inline void ocamDescrambler::mapped( short *__restrict__ dest, const short *__restrict__ raw ) const
{
    for( size_t i = 0; i < m_map.size(); ++i )
    {
        dest[i] = raw[m_map[i]];
    }
}

inline void ocamDescrambler::descramble( short *__restrict__ dest, const short *__restrict__ raw ) const
{
    if( !m_structured )
    {
        mapped( dest, raw );
        return;
    }

    for( size_t row = 0; row < m_rowBases.size(); ++row )
    {
        short      *output = dest + row * OCAM2_IMAGE_WIDTH_BINNING2x2;
        const auto &bases  = m_rowBases[row];

        for( int pixel = 0; pixel < 30; ++pixel )
        {
            output[pixel]      = raw[bases[0] + m_runStrides[0] * pixel];
            output[30 + pixel] = raw[bases[1] + m_runStrides[1] * pixel];
            output[60 + pixel] = raw[bases[2] + m_runStrides[2] * pixel];
            output[90 + pixel] = raw[bases[3] + m_runStrides[3] * pixel];
        }
    }
}

} // namespace app
} // namespace MagAOX

#endif // ocamDescrambler_hpp
