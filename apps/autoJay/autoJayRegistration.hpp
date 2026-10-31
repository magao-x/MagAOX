/** \file autoJayRegistration.hpp
 * \brief Registration and response-quality checks for paired DM images.
 * \ingroup autoJay_files
 */
#ifndef autoJayRegistration_hpp
#define autoJayRegistration_hpp

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <mx/improc/imageXCorrFFT.hpp>

#include "autoJayTypes.hpp"

namespace MagAOX
{
namespace app
{
namespace autoJayDetail
{

/// Acceptance thresholds in physical detector pixels and dimensionless quality units.
struct registrationSettings
{
    double maxShift{ 5 };              ///< Maximum magnitude of either individual displacement component.
    double maxDifferential{ 5 };       ///< Maximum magnitude of either differential component.
    double minCorrelation{ 0.5 };      ///< Minimum correlation on actual overlapping pixels.
    double minSnr{ 3 };                ///< Minimum excess spatial response RMS divided by stack noise RMS.
    double minOverlap{ 0.25 };         ///< Minimum fraction of reference pixels that overlap the live image.
    double minReferenceEnergy{ 0.25 }; ///< Minimum fraction of reference squared signal in the overlap.
    double maxSecondaryRatio{ 0.95 };  ///< Largest competing correlation divided by the primary correlation.
    double peakExclusion{ 2 };         ///< Radius around the principal peak excluded from ambiguity checking.
    double tolerance{ 0.05 };          ///< Requested subpixel localization sampling in detector pixels.
    double saturation{ std::numeric_limits<double>::max() }; ///< Raw mean saturation threshold, exclusive.
};

/// Registration of one baseline-subtracted response against its recorded template.
struct registrationResult
{
    bool        valid{ false };       ///< True only after all acceptance checks pass.
    std::string reason;               ///< Rejection explanation; empty for an accepted response.
    double      dx{ 0 };              ///< Horizontal displacement in detector pixels.
    double      dy{ 0 };              ///< Vertical displacement in detector pixels.
    double      correlation{ 0 };     ///< Pearson correlation over actual shifted overlap.
    double      snr{ 0 };             ///< Excess response RMS divided by mean-image noise RMS.
    double      overlap{ 0 };         ///< Fraction of reference pixels included in the score.
    double      referenceEnergy{ 0 }; ///< Fraction of reference squared signal included in the score.
    double      secondaryRatio{ 0 };  ///< Strongest supported competing peak relative to the principal peak.
};

/// Both registrations and their tweeter-minus-NCPC displacement.
struct pairResult
{
    bool                              valid{ false }; ///< True only for a complete accepted pair.
    std::string                       reason;         ///< Pair rejection explanation.
    std::array<registrationResult, 2> dm;             ///< NCPC first, tweeter second.
    double                            deltaX{ 0 };    ///< Horizontal calibrated differential in detector pixels.
    double                            deltaY{ 0 };    ///< Vertical calibrated differential in detector pixels.
};

/// Register a three-state acquisition to an aligned reference pair without publishing a sample.
inline pairResult
registerPair( const calibration               &reference,    /**< [in] Complete aligned reference pair. */
              const std::array<frameStack, 3> &frames,       /**< [in] Baseline, NCPC, and tweeter stacks. */
              const geometry                  &liveGeometry, /**< [in] Applied geometry shared by the three stacks. */
              const registrationSettings      &settings /**< [in] Registration and quality thresholds. */ );

/// Validate a first reference capture without depending on an older calibration.
inline pairResult
validateCapture( const std::array<frameStack, 3> &frames,         /**< [in] Baseline, NCPC, and tweeter stacks. */
                 const geometry                  &cameraGeometry, /**< [in] Applied geometry shared by all stacks. */
                 const registrationSettings      &settings /**< [in] Signal and ambiguity thresholds. */ );

namespace registrationDetail
{

/// One shared mapping of the reference ROI and live ROI into padded arrays.
struct mapping
{
    int    width{ 0 }, height{ 0 };          ///< Working array dimensions, with rows representing x.
    int    referenceX{ 0 }, referenceY{ 0 }; ///< Reference array placement.
    int    liveX{ 0 }, liveY{ 0 };           ///< Live array placement.
    double fractionX{ 0 }, fractionY{ 0 };   ///< Fractional live-minus-reference pixel-lattice offset.
    int    lag{ 0 };                         ///< Conservative sampled-pixel FFT search radius.
};

/// Validate geometry against an image's dimensions without assuming the ROI uses sensor units.
inline bool geometryValid( const geometry &value, /**< [in] Geometry to validate. */
                           const imageT   &image /**< [in] Image represented by the geometry. */ );

/// Validate finite configuration and meaningful threshold ranges.
inline bool settingsValid( const registrationSettings &settings /**< [in] Thresholds to validate. */ );

/// Reject incomplete, saturated, or nonfinite stacks before subtraction or normalization.
inline std::string validateFrames( const std::array<frameStack, 3> &frames,    /**< [in] Acquisition stacks. */
                                   const geometry             &cameraGeometry, /**< [in] Common applied geometry. */
                                   const registrationSettings &settings /**< [in] Saturation threshold. */ );

/// Construct a common array mapping while retaining fractional pixel-lattice offsets.
inline bool makeMapping( mapping                    &result,            /**< [out] Shared mapping. */
                         const geometry             &referenceGeometry, /**< [in] Reference pair geometry. */
                         const imageT               &reference,         /**< [in] Reference image dimensions. */
                         const geometry             &liveGeometry,      /**< [in] Live sequence geometry. */
                         const imageT               &live,              /**< [in] Live image dimensions. */
                         const registrationSettings &settings /**< [in] Physical search limit. */ );

/// Bilinearly sample an image inside its actual bounds.
inline double interpolate( const imageT &image, /**< [in] Image sampled inside its bounds. */
                           double        x,     /**< [in] Horizontal image coordinate. */
                           double        y /**< [in] Vertical image coordinate. */ );

/// Score one lag using only actual common support and propagated mean-image variance.
inline registrationResult
score( const imageT &reference, /**< [in] Unpadded reference response. */
       const imageT &live,      /**< [in] Unpadded live response. */
       const imageT &variance,  /**< [in] Live response variance of the mean. */
       double        localX,    /**< [in] Live-minus-reference index displacement. */
       double        localY,    /**< [in] Live-minus-reference index displacement. */
       const double *crossProduct = nullptr /**< [in] Optional FFT cross-product at an integer lag. */ );

/// Check overlap and reference signal support independently of correlation and noise.
inline bool supported( const registrationResult   &result, /**< [in] Candidate score. */
                       const registrationSettings &settings /**< [in] Required support fractions. */ );

/// Register one DM with mxlib, then validate overlap, response noise, and competing lags.
inline registrationResult
registerImage( const imageT               &reference,      /**< [in] Recorded response. */
               const imageT               &live,           /**< [in] Current baseline-subtracted response. */
               const imageT               &variance,       /**< [in] Propagated live variance of the mean. */
               const mapping              &map,            /**< [in] Shared ROI mapping. */
               const geometry             &cameraGeometry, /**< [in] Binning for physical shifts. */
               const registrationSettings &settings,       /**< [in] Acceptance thresholds. */
               bool refineVisible = true /**< [in] Permit one absent-feature rejection pass. */ );

inline bool geometryValid( const geometry &value, const imageT &image )
{
    return roi::valid( value.region ) && ( value.units == roi::units::sensor || value.units == roi::units::binned ) &&
           !value.orientation.empty() && image.rows() > 1 && image.cols() > 1 &&
           roi::imageSize( value.region.w, value.region.bin_x, value.units ) == image.rows() &&
           roi::imageSize( value.region.h, value.region.bin_y, value.units ) == image.cols();
}

inline bool settingsValid( const registrationSettings &s )
{
    for( double value : { s.maxShift,
                          s.maxDifferential,
                          s.minCorrelation,
                          s.minSnr,
                          s.minOverlap,
                          s.minReferenceEnergy,
                          s.maxSecondaryRatio,
                          s.peakExclusion,
                          s.tolerance,
                          s.saturation } )
    {
        if( !mx::math::isFinite( value ) )
            return false;
    }
    return s.maxShift > 0 && s.maxDifferential > 0 && s.minCorrelation > 0 && s.minCorrelation <= 1 && s.minSnr > 0 &&
           s.minOverlap > 0 && s.minOverlap <= 1 && s.minReferenceEnergy > 0 && s.minReferenceEnergy <= 1 &&
           s.maxSecondaryRatio > 0 && s.maxSecondaryRatio < 1 && s.peakExclusion > 0 && s.peakExclusion < s.maxShift &&
           s.tolerance > 0 && s.tolerance < s.maxShift && s.saturation > 0;
}

inline std::string validateFrames( const std::array<frameStack, 3> &frames,
                                   const geometry                  &cameraGeometry,
                                   const registrationSettings      &settings )
{
    for( const auto &frame : frames )
    {
        if( frame.count < 2 )
            return "at least two distinct frames are required to estimate noise";
        if( !geometryValid( cameraGeometry, frame.mean ) || frame.variance.rows() != frame.mean.rows() ||
            frame.variance.cols() != frame.mean.cols() )
            return "stack dimensions do not match applied geometry";
        for( Eigen::Index p = 0; p < frame.mean.size(); ++p )
        {
            if( !mx::math::isFinite( frame.mean.data()[p] ) || !mx::math::isFinite( frame.variance.data()[p] ) ||
                frame.variance.data()[p] < 0 )
                return "invalid stack pixels or noise variance";
            if( frame.mean.data()[p] >= settings.saturation )
                return "saturated stack";
        }
    }
    return {};
}

inline bool makeMapping( mapping                    &result,
                         const geometry             &rg,
                         const imageT               &reference,
                         const geometry             &lg,
                         const imageT               &live,
                         const registrationSettings &settings )
{
    if( !geometryValid( rg, reference ) || !geometryValid( lg, live ) || rg.orientation != lg.orientation ||
        rg.units != lg.units || rg.region.bin_x != lg.region.bin_x || rg.region.bin_y != lg.region.bin_y )
        return false;

    const double dx = ( roi::physical( lg.region.x, lg.region.bin_x, lg.units ) -
                        roi::physical( rg.region.x, rg.region.bin_x, rg.units ) ) /
                          rg.region.bin_x -
                      0.5 * ( live.rows() - reference.rows() );
    const double dy = ( roi::physical( lg.region.y, lg.region.bin_y, lg.units ) -
                        roi::physical( rg.region.y, rg.region.bin_y, rg.units ) ) /
                          rg.region.bin_y -
                      0.5 * ( live.cols() - reference.cols() );
    const double lagX = settings.maxShift / rg.region.bin_x;
    const double lagY = settings.maxShift / rg.region.bin_y;
    if( lagX > reference.rows() || lagY > reference.cols() )
        return false;
    if( dx >= reference.rows() + lagX || dx + live.rows() <= -lagX || dy >= reference.cols() + lagY ||
        dy + live.cols() <= -lagY )
        return false;

    // Real detector starts need not be separated by an integral number of binned samples.
    const int ix     = std::lround( dx );
    const int iy     = std::lround( dy );
    result.fractionX = dx - ix;
    result.fractionY = dy - iy;
    result.lag = std::ceil( std::max( lagX + std::abs( result.fractionX ), lagY + std::abs( result.fractionY ) ) );
    const int border  = result.lag + 2;
    result.referenceX = border - std::min( 0, ix );
    result.referenceY = border - std::min( 0, iy );
    result.liveX      = result.referenceX + ix;
    result.liveY      = result.referenceY + iy;
    result.width      = std::max( result.referenceX + reference.rows(), result.liveX + live.rows() ) + border;
    result.height     = std::max( result.referenceY + reference.cols(), result.liveY + live.cols() ) + border;
    return true;
}

inline double interpolate( const imageT &image, double x, double y )
{
    const int    x0 = std::floor( x ), y0 = std::floor( y );
    const int    x1 = std::min<int>( x0 + 1, image.rows() - 1 ), y1 = std::min<int>( y0 + 1, image.cols() - 1 );
    const double fx = x - x0, fy = y - y0;
    return ( 1 - fx ) * ( 1 - fy ) * image( x0, y0 ) + fx * ( 1 - fy ) * image( x1, y0 ) +
           ( 1 - fx ) * fy * image( x0, y1 ) + fx * fy * image( x1, y1 );
}

inline registrationResult score( const imageT &reference,
                                 const imageT &live,
                                 const imageT &variance,
                                 double        localX,
                                 double        localY,
                                 const double *crossProduct )
{
    registrationResult result;
    double             n = 0, sr = 0, st = 0, rr = 0, tt = 0, rt = 0, noise = 0;
    const double       totalEnergy = reference.template cast<double>().square().sum();
    for( Eigen::Index y = 0; y < reference.cols(); ++y )
    {
        for( Eigen::Index x = 0; x < reference.rows(); ++x )
        {
            const double lx = x + localX, ly = y + localY;
            if( lx < 0 || ly < 0 || lx > live.rows() - 1 || ly > live.cols() - 1 )
                continue;
            const double r = reference( x, y );
            const double t = interpolate( live, lx, ly );
            // Conservative interpolation of variance avoids claiming independent interpolation samples.
            noise += interpolate( variance, lx, ly );
            ++n;
            sr += r;
            st += t;
            rr += r * r;
            tt += t * t;
            if( crossProduct == nullptr )
                rt += r * t;
        }
    }
    result.overlap         = n / reference.size();
    result.referenceEnergy = totalEnergy > 0 ? rr / totalEnergy : 0;
    if( n < 4 )
        return result;
    rr -= sr * sr / n;
    tt -= st * st / n;
    if( rr <= 0 || tt <= 0 )
        return result;
    if( crossProduct != nullptr )
        rt = *crossProduct;
    result.correlation = std::clamp( ( rt - sr * st / n ) / std::sqrt( rr * tt ), -1.0, 1.0 );
    result.snr = noise > 0 ? std::sqrt( std::max( 0.0, ( tt - noise ) / noise ) ) : std::numeric_limits<double>::max();
    return result;
}

inline bool supported( const registrationResult &result, const registrationSettings &settings )
{
    return result.overlap >= settings.minOverlap && result.referenceEnergy >= settings.minReferenceEnergy;
}

inline registrationResult registerImage( const imageT               &reference,
                                         const imageT               &live,
                                         const imageT               &variance,
                                         const mapping              &map,
                                         const geometry             &cameraGeometry,
                                         const registrationSettings &settings,
                                         bool                        refineVisible )
{
    registrationResult result;
    for( Eigen::Index p = 0; p < reference.size(); ++p )
    {
        if( !mx::math::isFinite( reference.data()[p] ) )
        {
            result.reason = "nonfinite reference";
            return result;
        }
    }
    if( reference.size() == 0 || reference.maxCoeff() == reference.minCoeff() || live.maxCoeff() == live.minCoeff() )
    {
        result.reason = "zero response variance";
        return result;
    }
    for( Eigen::Index p = 0; p < live.size(); ++p )
    {
        if( !mx::math::isFinite( live.data()[p] ) || !mx::math::isFinite( variance.data()[p] ) )
        {
            result.reason = "nonfinite baseline-subtracted response or variance";
            return result;
        }
    }
    const int bx = cameraGeometry.region.bin_x, by = cameraGeometry.region.bin_y;
    const int ox = map.liveX - map.referenceX, oy = map.liveY - map.referenceY;
    // Guard samples permit cubic interpolation at the physical search boundary.
    const int radius  = map.lag + 2;
    imageT    surface = imageT::Constant( 2 * radius + 1, 2 * radius + 1, -1 );
    imageT    r = imageT::Zero( map.width, map.height ), t = r;
    r.block( map.referenceX, map.referenceY, reference.rows(), reference.cols() ) = reference;
    t.block( map.liveX, map.liveY, live.rows(), live.cols() )                     = live;
    try
    {
        mx::improc::imageXCorrFFT<imageT> correlator;
        correlator.normalize( false );
        correlator.peakMethod( mx::improc::xcorrPeakMethod::none );
        if( correlator.refIm( r ) < 0 )
            throw std::runtime_error( "reference transform failed" );
        float sx = 0, sy = 0, peak = 0;
        if( correlator( sx, sy, peak, t ) < 0 || !mx::math::isFinite( peak ) )
            throw std::runtime_error( "cross-correlation failed" );
        const int cx = std::lround( correlator.refX0() ), cy = std::lround( correlator.refY0() );
        for( int y = -radius; y <= radius; ++y )
        {
            for( int x = -radius; x <= radius; ++x )
            {
                const int    ix      = ( cx + x + map.width ) % map.width;
                const int    iy      = ( cy + y + map.height ) % map.height;
                const double product = correlator.ccIm()( ix, iy );
                // FFT supplies cross-products; overlap-specific centering and variances supply NCC.
                const auto candidate = score( reference, live, variance, x - ox, y - oy, &product );
                if( supported( candidate, settings ) && mx::math::isFinite( candidate.correlation ) )
                    surface( x + radius, y + radius ) = candidate.correlation;
            }
        }
    }
    catch( const std::exception &error )
    {
        result.reason = std::string( "registration failed: " ) + error.what();
        return result;
    }
    const double subdivisions = std::ceil( 2 * radius * std::max( bx, by ) / settings.tolerance );
    if( subdivisions > 4096 )
    {
        result.reason = "registration precision requires an excessive working grid";
        return result;
    }
    const int    length = static_cast<int>( subdivisions ) + 1;
    const double step   = 2.0 * radius / ( length - 1 );
    imageT       fine( length, length );
    mx::improc::imageMagnify( fine, surface, mx::improc::cubicConvolTransform<double>() );
    double best  = -2;
    int    bestX = 0, bestY = 0;
    for( int y = 0; y < length; ++y )
    {
        for( int x = 0; x < length; ++x )
        {
            const double dx = ( x * step - radius + map.fractionX ) * bx;
            const double dy = ( y * step - radius + map.fractionY ) * by;
            if( std::abs( dx ) <= settings.maxShift && std::abs( dy ) <= settings.maxShift && fine( x, y ) > best )
            {
                best  = fine( x, y );
                bestX = x;
                bestY = y;
            }
        }
    }
    const double sx = bestX * step - radius, sy = bestY * step - radius;
    result    = score( reference, live, variance, sx - ox, sy - oy );
    result.dx = ( sx + map.fractionX ) * bx;
    result.dy = ( sy + map.fractionY ) * by;
    if( std::abs( result.dx ) >= settings.maxShift - settings.tolerance ||
        std::abs( result.dy ) >= settings.maxShift - settings.tolerance )
        result.reason = "individual displacement exceeds the accepted search interior";
    else if( !supported( result, settings ) )
        result.reason = "insufficient overlapping reference support";
    else if( result.correlation < settings.minCorrelation )
        result.reason = "low or inverted response correlation";
    else if( result.snr < settings.minSnr )
        result.reason = "low response SNR";
    if( !result.reason.empty() )
        return result;

    // A fitted visibility mask cannot resolve ambiguity already present in the original reference.
    double second = 0;
    for( int y = 0; y < length; ++y )
    {
        for( int x = 0; x < length; ++x )
        {
            const double dx = ( x * step - radius + map.fractionX ) * bx;
            const double dy = ( y * step - radius + map.fractionY ) * by;
            if( std::abs( dx ) <= settings.maxShift && std::abs( dy ) <= settings.maxShift &&
                std::hypot( dx - result.dx, dy - result.dy ) > settings.peakExclusion )
                second = std::max<double>( second, fine( x, y ) );
        }
    }
    result.secondaryRatio = second / best;
    if( result.secondaryRatio >= settings.maxSecondaryRatio )
        result.reason = "ambiguous competing registration peak";
    if( !result.reason.empty() )
        return result;

    if( refineVisible )
    {
        imageT       visible                = reference;
        const double floor                  = 0.01 * reference.abs().maxCoeff();
        const double totalEnergy            = reference.template cast<double>().square().sum();
        double       referenceOverlapEnergy = 0, liveOverlapEnergy = 0;
        for( Eigen::Index y = 0; y < reference.cols(); ++y )
        {
            for( Eigen::Index x = 0; x < reference.rows(); ++x )
            {
                const double lx = x + sx - ox, ly = y + sy - oy;
                if( lx < 0 || ly < 0 || lx > live.rows() - 1 || ly > live.cols() - 1 )
                    continue;
                const double observed = interpolate( live, lx, ly );
                referenceOverlapEnergy += double( reference( x, y ) ) * reference( x, y );
                liveOverlapEnergy += observed * observed;
            }
        }
        const double gain    = std::sqrt( liveOverlapEnergy / referenceOverlapEnergy );
        double       removed = 0;
        for( Eigen::Index y = 0; y < reference.cols(); ++y )
        {
            for( Eigen::Index x = 0; x < reference.rows(); ++x )
            {
                const double lx = x + sx - ox, ly = y + sy - oy;
                const double value = reference( x, y );
                if( lx < 0 || ly < 0 || lx > live.rows() - 1 || ly > live.cols() - 1 || std::abs( value ) < floor )
                    continue;
                if( std::abs( interpolate( live, lx, ly ) ) < 0.2 * gain * std::abs( value ) )
                {
                    removed += value * value;
                    visible( x, y ) = 0;
                }
            }
        }
        // Do not let tiny noisy tail pixels trigger a second registration or alter its support accounting.
        if( removed > 0.02 * totalEnergy )
        {
            const double retained = ( totalEnergy - removed ) / totalEnergy;
            if( retained < settings.minReferenceEnergy )
            {
                result.valid  = false;
                result.reason = "insufficient reference signal remains visible";
                return result;
            }
            result = registerImage( visible, live, variance, map, cameraGeometry, settings, false );
            result.referenceEnergy *= retained;
            if( result.referenceEnergy < settings.minReferenceEnergy )
            {
                result.valid  = false;
                result.reason = "insufficient overlapping reference signal remains visible";
            }
            return result;
        }
    }

    result.valid = true;
    return result;
}

} // namespace registrationDetail

inline pairResult registerPair( const calibration               &reference,
                                const std::array<frameStack, 3> &frames,
                                const geometry                  &liveGeometry,
                                const registrationSettings      &settings )
{
    pairResult result;
    if( !registrationDetail::settingsValid( settings ) )
    {
        result.reason = "invalid registration settings";
        return result;
    }
    result.reason = registrationDetail::validateFrames( frames, liveGeometry, settings );
    if( !result.reason.empty() )
        return result;
    registrationDetail::mapping map;
    if( reference.dm[0].reference.rows() != reference.dm[1].reference.rows() ||
        reference.dm[0].reference.cols() != reference.dm[1].reference.cols() ||
        !registrationDetail::makeMapping(
            map, reference.cameraGeometry, reference.dm[0].reference, liveGeometry, frames[0].mean, settings ) )
    {
        result.reason = "incompatible reference/live geometry or no overlap";
        return result;
    }
    for( size_t dm = 0; dm < 2; ++dm )
    {
        imageT response = frames[dm + 1].mean - frames[0].mean;
        imageT variance = frames[dm + 1].variance + frames[0].variance;
        result.dm[dm]   = registrationDetail::registerImage(
            reference.dm[dm].reference, response, variance, map, liveGeometry, settings );
        if( !result.dm[dm].valid )
        {
            result.reason = std::string( dm == 0 ? "NCPC: " : "tweeter: " ) + result.dm[dm].reason;
            return result;
        }
    }
    result.deltaX = result.dm[1].dx - result.dm[0].dx;
    result.deltaY = result.dm[1].dy - result.dm[0].dy;
    if( std::abs( result.deltaX ) > settings.maxDifferential || std::abs( result.deltaY ) > settings.maxDifferential )
        result.reason = "differential displacement exceeds limit";
    else
        result.valid = true;
    return result;
}

inline pairResult validateCapture( const std::array<frameStack, 3> &frames,
                                   const geometry                  &cameraGeometry,
                                   const registrationSettings      &settings )
{
    pairResult result;
    if( !registrationDetail::settingsValid( settings ) )
    {
        result.reason = "invalid registration settings";
        return result;
    }
    result.reason = registrationDetail::validateFrames( frames, cameraGeometry, settings );
    if( !result.reason.empty() )
        return result;
    calibration reference;
    reference.cameraGeometry = cameraGeometry;
    for( size_t dm = 0; dm < 2; ++dm )
        reference.dm[dm].reference = frames[dm + 1].mean - frames[0].mean;
    return registerPair( reference, frames, cameraGeometry, settings );
}

} // namespace autoJayDetail
} // namespace app
} // namespace MagAOX

#endif // autoJayRegistration_hpp
