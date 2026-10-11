/** \file roiGeometry.hpp
 * \brief Camera ROI coordinates shared by camera applications and display clients.
 */
#ifndef roiGeometry_hpp
#define roiGeometry_hpp

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace MagAOX
{
namespace roi
{
/// Units of the unchanged, full-frame camera ROI interface.
enum class units
{
    sensor      = 0, ///< Centers and extents before binning.
    binned      = 1, ///< Centers and extents in the full image at the selected binning.
    unsupported = 2  ///< No verified image-to-camera mapping.
};

/// A camera region in its native full-frame coordinates.
struct region
{
    float x{ 0 };     ///< Horizontal pixel-center coordinate.
    float y{ 0 };     ///< Vertical pixel-center coordinate.
    int   w{ 0 };     ///< Width in the camera's ROI units.
    int   h{ 0 };     ///< Height in the camera's ROI units.
    int   bin_x{ 0 }; ///< Detector pixels combined horizontally.
    int   bin_y{ 0 }; ///< Detector pixels combined vertically.
};

/// Check finite centers and positive, representable extents and binning.
inline bool valid( const region &r /**< [in] Region to validate. */ );

/// Compare native ROI values without rounding their displayed representation.
inline bool same( const region &a, /**< [in] First region. */
                  const region &b /**< [in] Second region. */ );

/// Convert a native center to a physical detector-pixel center.
inline double physical( double center, /**< [in] Native center. */
                        int    bin,    /**< [in] Binning for this center. */
                        units  u /**< [in] Native coordinate convention. */ );

/// Get the image length represented by one native extent.
inline double imageSize( int   extent, /**< [in] Native extent. */
                         int   bin,    /**< [in] Current binning. */
                         units u /**< [in] Native coordinate convention. */ );

/// Project a target center into the currently acquired image.
inline double relative( double target,    /**< [in] Native target center. */
                        int    targetBin, /**< [in] Target binning. */
                        double current,   /**< [in] Native applied center. */
                        int    extent,    /**< [in] Applied native extent. */
                        int    bin,       /**< [in] Applied binning. */
                        units  u /**< [in] Native coordinate convention. */ );

/// Convert a current-image center into a native full-frame target center.
inline double absolute( double relativeCenter, /**< [in] Current-image center. */
                        int    targetBin,      /**< [in] Target binning. */
                        double current,        /**< [in] Native applied center. */
                        int    extent,         /**< [in] Applied native extent. */
                        int    bin,            /**< [in] Applied binning. */
                        units  u /**< [in] Native coordinate convention. */ );

/// Return the six native ROI values in protocol order.
inline std::vector<double> values( const region &r /**< [in] Region to serialize. */ );

/// Return the six ROI element names in protocol order.
inline std::vector<std::string> names( const std::string &prefix = "" /**< [in] Optional element prefix. */ );

inline bool valid( const region &r )
{
    // Camera applications use -ffast-math; inspect IEEE exponent bits instead of an optimized-away isfinite.
    static_assert(sizeof(float) == sizeof(uint32_t) && std::numeric_limits<float>::is_iec559);
    uint32_t x, y;
    std::memcpy(&x, &r.x, sizeof(x));
    std::memcpy(&y, &r.y, sizeof(y));
    return (x & 0x7f800000u) != 0x7f800000u && (y & 0x7f800000u) != 0x7f800000u &&
           r.w > 0 && r.h > 0 && r.bin_x > 0 && r.bin_y > 0;
}

inline bool same( const region &a, const region &b )
{
    return a.x == b.x && a.y == b.y && a.w == b.w && a.h == b.h && a.bin_x == b.bin_x && a.bin_y == b.bin_y;
}

inline double physical( double center, int bin, units u )
{
    return u == units::binned ? bin * ( center + 0.5 ) - 0.5 : center;
}

inline double imageSize( int extent, int bin, units u )
{
    if( bin <= 0 || extent <= 0 || u == units::unsupported )
        return std::numeric_limits<double>::quiet_NaN();
    return u == units::binned ? extent : static_cast<double>( extent ) / bin;
}

inline double relative( double target, int targetBin, double current, int extent, int bin, units u )
{
    return ( physical( target, targetBin, u ) - physical( current, bin, u ) ) / bin +
           ( imageSize( extent, bin, u ) - 1 ) / 2;
}

inline double absolute( double relativeCenter, int targetBin, double current, int extent, int bin, units u )
{
    double center = physical( current, bin, u ) + bin * ( relativeCenter - ( imageSize( extent, bin, u ) - 1 ) / 2 );
    return u == units::binned ? ( center + 0.5 ) / targetBin - 0.5 : center;
}

inline std::vector<double> values( const region &r )
{
    return { r.x,
             r.y,
             static_cast<double>( r.w ),
             static_cast<double>( r.h ),
             static_cast<double>( r.bin_x ),
             static_cast<double>( r.bin_y ) };
}

inline std::vector<std::string> names( const std::string &prefix )
{
    return { prefix + "x", prefix + "y", prefix + "w", prefix + "h", prefix + "bin_x", prefix + "bin_y" };
}

} // namespace roi
} // namespace MagAOX
#endif
