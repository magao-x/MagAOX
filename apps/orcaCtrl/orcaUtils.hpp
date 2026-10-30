/** \file orcaUtils.hpp
 * \brief Utilities for the Hamamatsu ORCA-Quest2 camera controller.
 *
 * These helpers have no DCAM dependency so they can be unit tested without the camera SDK.
 *
 * \author Joshua Liberman (jliberman54@gmail.com)
 *
 * \ingroup orcaCtrl_files
 */

#ifndef orcaUtils_hpp
#define orcaUtils_hpp

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace MagAOX
{
namespace app
{

/// The DCAM ring-buffer frame chosen for publishing by orcaSelectFrame().
/**
 * \ingroup orcaCtrl
 */
struct orcaFrameSelection
{
    int64_t frame{ 0 };     ///< Frame number to publish, counted from the start of capture.
    int32_t ringIndex{ 0 }; ///< Index of that frame in the DCAM ring buffer.
    int64_t dropped{ 0 };   ///< Frames skipped because the ring buffer overwrote them before publishing.
};

/// Choose the next frame to publish from the DCAM ring buffer.
/** Frames are published in order. If the backlog of unpublished frames is larger than the ring buffer
 * can safely hold (`ringSize - margin`), the oldest frames have been or are being overwritten by DCAM, so
 * the selection skips ahead to the oldest frame that is still safe and reports how many were dropped.
 *
 * The ring index is computed relative to the newest transferred frame, so no assumption is made about
 * which ring index capture started at.
 *
 * \returns 0 on success
 * \returns -1 if the inputs are invalid (no unpublished frame, bad ring size, margin or newest index)
 *
 * \ingroup orcaCtrl
 */
inline int orcaSelectFrame( orcaFrameSelection &sel,               /**< [out] the frame to publish */
                            int64_t             nextFrame,         /**< [in] next frame number not yet published */
                            int64_t             framesTransferred, /**< [in] total frames DCAM has transferred */
                            int32_t             newestIndex,       /**< [in] ring index of the newest transferred
                                                                             frame (frame `framesTransferred - 1`) */
                            int32_t ringSize,                      /**< [in] number of frames in the ring buffer */
                            int32_t margin                         /**< [in] ring slots treated as unsafe because
                                                                             DCAM may be writing them */
)
{
    if( nextFrame < 0 || nextFrame >= framesTransferred || ringSize <= 0 || margin < 0 || margin >= ringSize ||
        newestIndex < 0 || newestIndex >= ringSize )
    {
        return -1;
    }

    const int64_t maxBacklog = ringSize - margin;

    sel.frame   = nextFrame;
    sel.dropped = 0;

    if( framesTransferred - nextFrame > maxBacklog )
    {
        sel.frame   = framesTransferred - maxBacklog;
        sel.dropped = sel.frame - nextFrame;
    }

    const int64_t behind = ( framesTransferred - 1 ) - sel.frame; // how many frames older than the newest

    sel.ringIndex = static_cast<int32_t>( ( ( newestIndex - behind ) % ringSize + ringSize ) % ringSize );

    return 0;
}

/// Check that a DCAM framestamp advanced by the expected number of frames.
/** The comparison is modulo 2^16. On the ORCA-Quest2 the framestamp behaves as a 16-bit counter:
 * hardware tests (agents/plans/2026-10/orcaCtrl-latency-new.md) showed about one mismatch per 65,536 frames
 * when compared as a full 32-bit value. Comparing the low 16 bits also works for a 32-bit counter.
 *
 * \returns true if the framestamp advanced by `frames`, modulo 2^16
 * \returns false otherwise
 *
 * \ingroup orcaCtrl
 */
inline bool orcaFramestampAdvanceOk( int32_t lastStamp, /**< [in] framestamp of the earlier frame */
                                     int32_t stamp,     /**< [in] framestamp of the later frame */
                                     int64_t frames     /**< [in] frame-number difference between them */
)
{
    const uint32_t stampDelta = static_cast<uint32_t>( stamp ) - static_cast<uint32_t>( lastStamp );

    return ( stampDelta & 0xFFFFu ) == ( static_cast<uint64_t>( frames ) & 0xFFFFu );
}

/// Number of DCAM ring-buffer frames to allocate.
/** Sizes the buffer to hold at least `minSeconds` of frames at the current frame rate, and at least
 * `minFrames`, then limits it to `maxBytes` of memory and to `maxFrames`. The result is never less than
 * 3 frames, which DCAM needs for continuous capture, even if that exceeds `maxBytes`.
 *
 * \returns the number of frames to pass to dcambuf_alloc()
 *
 * \ingroup orcaCtrl
 */
inline int32_t orcaBufferFrames( uint64_t frameBytes, /**< [in] bytes per frame (DCAM_IDPROP_IMAGE_FRAMEBYTES) */
                                 double   fps,        /**< [in] frame rate [Hz] */
                                 double   minSeconds, /**< [in] minimum time the buffer should hold [s] */
                                 uint64_t maxBytes,   /**< [in] maximum buffer memory [bytes] */
                                 int32_t  minFrames,  /**< [in] minimum number of frames */
                                 int32_t  maxFrames   /**< [in] maximum number of frames */
)
{
    double want = minFrames;

    if( fps > 0 && minSeconds > 0 )
    {
        want = std::max( want, std::ceil( fps * minSeconds ) );
    }

    want = std::min( want, static_cast<double>( maxFrames ) );

    if( frameBytes > 0 )
    {
        want = std::min( want, static_cast<double>( maxBytes / frameBytes ) );
    }

    return static_cast<int32_t>( std::max( want, 3.0 ) );
}

/// Round a requested ROI to the camera's size and position steps and clamp it onto the sensor.
/** The ROI is given as a center and a size, as used by stdCamera. The corner is computed as
 * `int( center ) - ( size - 1 ) / 2`, the same way orcaCtrl::setDcamRoi() does. Sizes are rounded down to
 * a multiple of `step` (at least one step, at most the sensor size). Corners are clamped onto the
 * sensor and rounded to the nearest multiple of `step`. The center is then recomputed from the corner.
 *
 * The ORCA-Quest2 uses 4-pixel size and position steps in area readout mode (manual, section 11-1-2).
 *
 * \returns true if any value was changed
 * \returns false if the ROI was already valid
 *
 * \ingroup orcaCtrl
 */
inline bool orcaClampROI( float  &xCen,    /**< [in,out] ROI center x [pixels] */
                          float  &yCen,    /**< [in,out] ROI center y [pixels] */
                          int    &w,       /**< [in,out] ROI width [pixels] */
                          int    &h,       /**< [in,out] ROI height [pixels] */
                          int32_t sensorW, /**< [in] sensor width [pixels] */
                          int32_t sensorH, /**< [in] sensor height [pixels] */
                          int32_t step     /**< [in] size and position step [pixels] */
)
{
    const float xIn = xCen;
    const float yIn = yCen;
    const int   wIn = w;
    const int   hIn = h;

    auto clampSize = [step]( int size, int32_t sensorSize ) -> int
    {
        size = ( size / step ) * step;
        return std::min( std::max( size, static_cast<int>( step ) ), static_cast<int>( sensorSize ) );
    };

    auto clampCorner = [step]( float center, int size, int32_t sensorSize ) -> float
    {
        int corner = static_cast<int>( center ) - ( size - 1 ) / 2;
        corner     = std::min( std::max( corner, 0 ), static_cast<int>( sensorSize ) - size );
        corner     = ( ( corner + step / 2 ) / step ) * step;
        corner     = std::min( corner, static_cast<int>( sensorSize ) - size );
        return static_cast<float>( corner + ( size - 1 ) / 2 );
    };

    w    = clampSize( w, sensorW );
    h    = clampSize( h, sensorH );
    xCen = clampCorner( xCen, w, sensorW );
    yCen = clampCorner( yCen, h, sensorH );

    return ( xCen != xIn || yCen != yIn || w != wIn || h != hIn );
}

} // namespace app
} // namespace MagAOX

#endif // orcaUtils_hpp
