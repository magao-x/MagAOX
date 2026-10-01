/** \file dmTemporalResponse.hpp
 * \brief The MagAO-X DM temporal response measurement application header.
 *
 * \author Katie Twitchell (twitchell@arizona.edu)
 *
 * \ingroup dmTemporalResponse_files
 */

#ifndef dmTemporalResponse_hpp
#define dmTemporalResponse_hpp

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "../../libMagAOX/libMagAOX.hpp" // Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

/** \defgroup dmTemporalResponse
 * \brief The MagAO-X application to measure the temporal response of a DM with sub-frame resolution.
 *
 * Pokes are triggered off the camWFS frame semaphore, delayed by a grid of sub-frame delays referenced to the frame
 * acquisition time (`atime`), and recorded as averaged +/- difference cubes plus scalar response metrics.
 *
 * See `agents/plans/2026-09-30/dm_response_merged_plan.md` for the design record.
 *
 * <a href="../handbook/operating/software/apps/dmTemporalResponse.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup dmTemporalResponse_files
 * \ingroup dmTemporalResponse
 */

namespace MagAOX
{
namespace app
{

/// Pure helper functions for dmTemporalResponse.
/** These have no hardware or INDI dependencies so they can be unit tested directly.
 *
 * \ingroup dmTemporalResponse
 */
namespace dmTemporalResponseMath
{

/// The two poke modes supported by the app.
enum class pokeMode
{
    actuator, ///< Poke a single actuator at (x, y).
    pattern   ///< Apply a 2-D DM pattern loaded from a FITS file.
};

/// Per-delay response metrics.
struct responseMetrics
{
    double m_t50{ std::numeric_limits<double>::quiet_NaN() }; ///< Time [us] from the DM command to r = 0.5.

    double m_rise{ std::numeric_limits<double>::quiet_NaN() }; ///< 10-90% rise time [us].

    double m_overshoot{ std::numeric_limits<double>::quiet_NaN() }; ///< max(r) - 1.

    double m_settleErr{ std::numeric_limits<double>::quiet_NaN() }; ///< RMS of r - 1 over the trailing frames.

    double m_jitter{ std::numeric_limits<double>::quiet_NaN() }; ///< Trial-to-trial std of r at the t50 frame.

    double m_delayErrMean{ std::numeric_limits<double>::quiet_NaN() }; ///< Mean of achieved - requested delay [us].

    double m_delayErrStd{ std::numeric_limits<double>::quiet_NaN() }; ///< Std of achieved - requested delay [us].

    double m_lateFrac{ std::numeric_limits<double>::quiet_NaN() }; ///< Fraction of trials whose poke deadline had passed.
};

/// Get the current CLOCK_REALTIME time.
/** This is the default clock used for the poke busy-wait and the command timestamp.
 *
 * \returns the current time
 */
inline timespec realtimeNow()
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    return ts;
}

/// Compute the difference a - b in microseconds.
/**
 * \returns a - b in microseconds
 */
inline double tsDiffUs( const timespec &a, /**< [in] the later time */
                        const timespec &b  /**< [in] the earlier time */
)
{
    return static_cast<double>( a.tv_sec - b.tv_sec ) * 1e6 + static_cast<double>( a.tv_nsec - b.tv_nsec ) * 1e-3;
}

/// Add a (possibly fractional, non-negative) number of microseconds to a timespec.
/**
 * \returns the sum
 */
inline timespec tsAddUs( const timespec &ts, /**< [in] the starting time */
                         double          us  /**< [in] microseconds to add, >= 0 */
)
{
    timespec out  = ts;
    int64_t  nsec = static_cast<int64_t>( ts.tv_nsec ) + static_cast<int64_t>( std::llround( us * 1e3 ) );

    out.tv_sec += static_cast<time_t>( nsec / 1000000000LL );
    out.tv_nsec = static_cast<long>( nsec % 1000000000LL );

    return out;
}

/// Build the DM channel stream name `dm<NN>disp<MM>`.
/**
 * \returns 0 on success
 * \returns -1 if \p nn is not 0-2 or \p mm is not 0-99
 */
inline int dmStreamName( std::string &name, /**< [out] the stream name, e.g. dm00disp07 */
                         int          nn,   /**< [in] the DM index: 0 woofer, 1 tweeter, 2 NCPC */
                         int          mm    /**< [in] the dmcomb channel number */
)
{
    if( nn < 0 || nn > 2 || mm < 0 || mm > 99 )
    {
        return -1;
    }

    char buf[32];
    snprintf( buf, sizeof( buf ), "dm%02ddisp%02d", nn, mm );
    name = buf;

    return 0;
}

/// Parse the poke mode from its string name.
/**
 * \returns 0 on success
 * \returns -1 if \p str is not "actuator" or "pattern"
 */
inline int parsePokeMode( pokeMode          &mode, /**< [out] the parsed mode */
                          const std::string &str   /**< [in] "actuator" or "pattern" */
)
{
    if( str == "actuator" )
    {
        mode = pokeMode::actuator;
        return 0;
    }

    if( str == "pattern" )
    {
        mode = pokeMode::pattern;
        return 0;
    }

    return -1;
}

/// Validate a single-actuator poke specification.
/**
 * \returns 0 if exactly one in-bounds actuator is specified
 * \returns -1 otherwise
 */
inline int validateActuator( const std::vector<int> &x,    /**< [in] x coordinates (must have exactly one entry) */
                             const std::vector<int> &y,    /**< [in] y coordinates (must have exactly one entry) */
                             uint32_t                rows, /**< [in] DM channel rows (x dimension) */
                             uint32_t                cols  /**< [in] DM channel cols (y dimension) */
)
{
    if( x.size() != 1 || y.size() != 1 )
    {
        return -1;
    }

    if( x[0] < 0 || y[0] < 0 || static_cast<uint32_t>( x[0] ) >= rows || static_cast<uint32_t>( y[0] ) >= cols )
    {
        return -1;
    }

    return 0;
}

/// Validate the poke amplitude against the command safety limit.
/**
 * \returns 0 if the amplitude is non-zero, finite, and |amp| <= maxCommand
 * \returns -1 otherwise
 */
inline int validateCommand( float amp,       /**< [in] the poke amplitude */
                            float maxCommand /**< [in] the maximum absolute DM command allowed */
)
{
    if( !std::isfinite( amp ) || amp == 0 || std::fabs( amp ) > maxCommand )
    {
        return -1;
    }

    return 0;
}

/// Validate a loaded DM pattern.
/**
 * \returns 0 if all values are finite, the pattern is not all zero, and max|amp*pattern| <= maxCommand
 * \returns -1 otherwise
 */
inline int validatePattern( const mx::improc::eigenImage<float> &pattern,   /**< [in] the pattern to validate */
                            float                                amp,       /**< [in] the poke amplitude */
                            float                                maxCommand /**< [in] the command safety limit */
)
{
    if( pattern.size() == 0 )
    {
        return -1;
    }

    float maxAbs = 0;

    for( int cc = 0; cc < pattern.cols(); ++cc )
    {
        for( int rr = 0; rr < pattern.rows(); ++rr )
        {
            if( !std::isfinite( pattern( rr, cc ) ) )
            {
                return -1;
            }

            maxAbs = std::max( maxAbs, std::fabs( pattern( rr, cc ) ) );
        }
    }

    if( maxAbs == 0 )
    {
        return -1;
    }

    if( std::fabs( amp ) * maxAbs > maxCommand )
    {
        return -1;
    }

    return 0;
}

/// Load a 2-D DM pattern from a FITS file and check its dimensions.
/**
 * \returns 0 on success
 * \returns -1 if the file can not be read, is not a single 2-D image, or has the wrong dimensions
 */
inline int loadPattern( mx::improc::eigenImage<float> &pattern, /**< [out] the loaded pattern */
                        const std::string             &path,    /**< [in] path to the FITS file */
                        uint32_t                       rows,    /**< [in] required rows (DM x dimension) */
                        uint32_t                       cols     /**< [in] required cols (DM y dimension) */
)
{
    if( path == "" || !std::filesystem::exists( path ) )
    {
        return -1;
    }

    mx::improc::eigenCube<float>                          cube;
    mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY>      ff;

    try
    {
        if( ff.read( cube, path ) != mx::error_t::noerror )
        {
            return -1;
        }
    }
    catch( ... )
    {
        return -1;
    }

    if( cube.planes() != 1 )
    {
        return -1;
    }

    if( static_cast<uint32_t>( cube.rows() ) != rows || static_cast<uint32_t>( cube.cols() ) != cols )
    {
        return -1;
    }

    pattern = cube.image( 0 );

    return 0;
}

/// Build a poke command for either mode.
/** In actuator mode the command is sign*amp at (x, y) and zero elsewhere.
 * In pattern mode the command is sign*amp*pattern (the pattern is not normalized).
 *
 * \returns 0 on success
 * \returns -1 if the inputs are inconsistent
 */
inline int buildPokeCommand( mx::improc::eigenImage<float>       &cmd,     /**< [out] the command, pre-sized */
                             pokeMode                             mode,    /**< [in] the poke mode */
                             int                                  x,       /**< [in] actuator x (actuator mode) */
                             int                                  y,       /**< [in] actuator y (actuator mode) */
                             const mx::improc::eigenImage<float> &pattern, /**< [in] the pattern (pattern mode) */
                             int                                  sign,    /**< [in] +1 or -1 */
                             float                                amp      /**< [in] the poke amplitude */
)
{
    if( mode == pokeMode::pattern )
    {
        if( pattern.rows() != cmd.rows() || pattern.cols() != cmd.cols() )
        {
            return -1;
        }

        cmd = pattern * ( sign * amp );

        return 0;
    }

    if( x < 0 || y < 0 || x >= cmd.rows() || y >= cmd.cols() )
    {
        return -1;
    }

    cmd.setZero();
    cmd( x, y ) = sign * amp;

    return 0;
}

/// Resolve the span of the delay grid.
/**
 * \returns 0 on success
 * \returns -1 if the default span is requested but the frame rate is unknown
 */
inline int resolveSpan( double &span,      /**< [out] the resolved span [us] */
                        double  delaySpan, /**< [in] the configured span [us], <= 0 means one frame period */
                        double  fps        /**< [in] the camWFS frame rate [Hz], <= 0 if unknown */
)
{
    if( delaySpan > 0 )
    {
        span = delaySpan;
        return 0;
    }

    if( fps <= 0 )
    {
        return -1;
    }

    span = 1e6 / fps;

    return 0;
}

/// Build the evenly spaced delay grid d_k = k*span/K, k = 0..K-1 (endpoint excluded).
/**
 * \returns 0 on success
 * \returns -1 if nDelays < 1 or span <= 0
 */
inline int delayGrid( std::vector<double> &delays,  /**< [out] the delays [us] */
                      int                  nDelays, /**< [in] K, the number of delays */
                      double               span     /**< [in] the span of the grid [us] */
)
{
    if( nDelays < 1 || span <= 0 )
    {
        return -1;
    }

    delays.resize( nDelays );

    for( int k = 0; k < nDelays; ++k )
    {
        delays[k] = k * span / nDelays;
    }

    return 0;
}

/// Validate the number of trials per delay.
/**
 * \returns 0 if nTrials is even and >= 2
 * \returns -1 otherwise
 */
inline int validateTrials( int nTrials /**< [in] M, the total number of trials per delay */ )
{
    if( nTrials < 2 || nTrials % 2 != 0 )
    {
        return -1;
    }

    return 0;
}

/// Format the UTC run directory name, YYYY-MM-DDTHHMMSS.
/**
 * \returns the directory name
 */
inline std::string runDirName( const timespec &ts /**< [in] the run start time */ )
{
    tm     uttime;
    time_t sec = ts.tv_sec;

    gmtime_r( &sec, &uttime );

    char buf[32];
    strftime( buf, sizeof( buf ), "%Y-%m-%dT%H%M%S", &uttime );

    return buf;
}

/// Format the ISO 8601 UTC date string for DATE-OBS.
/**
 * \returns the date string, YYYY-MM-DDTHH:MM:SS
 */
inline std::string isoDate( const timespec &ts /**< [in] the time to format */ )
{
    tm     uttime;
    time_t sec = ts.tv_sec;

    gmtime_r( &sec, &uttime );

    char buf[32];
    strftime( buf, sizeof( buf ), "%Y-%m-%dT%H:%M:%S", &uttime );

    return buf;
}

/// Format the per-delay cube file name, dmresp_delay_<DDDDD>us.fits.
/**
 * \returns the file name, with the delay rounded to the nearest microsecond
 */
inline std::string cubeFileName( double delay /**< [in] the delay [us] */ )
{
    char buf[64];
    snprintf( buf, sizeof( buf ), "dmresp_delay_%05lldus.fits", static_cast<long long>( std::llround( delay ) ) );

    return buf;
}

/// Compute the achieved delay of a poke.
/**
 * \returns tCmd - trigATime in microseconds (negative values are reported, not clamped)
 */
inline double achievedDelay( const timespec &trigATime, /**< [in] the trigger frame acquisition time */
                             const timespec &tCmd       /**< [in] the time the DM command was written */
)
{
    return tsDiffUs( tCmd, trigATime );
}

/// Form the +/- difference cube (sum+ - sum-)/M = (mean+ - mean-)/2.
/**
 * \returns 0 on success
 * \returns -1 if the sums have different sizes or M < 2
 */
inline int differenceCube( mx::improc::eigenCube<float>        &out,    /**< [out] the difference cube */
                           const mx::improc::eigenCube<double> &sumPos, /**< [in] sum of the M/2 + trials */
                           const mx::improc::eigenCube<double> &sumNeg, /**< [in] sum of the M/2 - trials */
                           int                                  M       /**< [in] the total number of trials */
)
{
    if( sumPos.rows() != sumNeg.rows() || sumPos.cols() != sumNeg.cols() || sumPos.planes() != sumNeg.planes() ||
        M < 2 )
    {
        return -1;
    }

    out.resize( sumPos.rows(), sumPos.cols(), sumPos.planes() );

    size_t n = static_cast<size_t>( sumPos.rows() ) * sumPos.cols() * sumPos.planes();

    for( size_t i = 0; i < n; ++i )
    {
        out.data()[i] = static_cast<float>( ( sumPos.data()[i] - sumNeg.data()[i] ) / M );
    }

    return 0;
}

/// Build the reference pixel mask |P| > thresh*max|P| and the projection normalization.
/**
 * \returns 0 on success
 * \returns -1 if P is empty or all zero
 */
inline int buildMask( mx::improc::eigenImage<float>       &mask,  /**< [out] the mask, 1 inside and 0 outside */
                      double                              &norm,  /**< [out] sum over the mask of P^2 */
                      const mx::improc::eigenImage<float> &P,     /**< [in] the reference pattern */
                      double                               thresh /**< [in] fraction of max|P| */
)
{
    if( P.size() == 0 )
    {
        return -1;
    }

    double maxAbs = P.abs().maxCoeff();

    if( !( maxAbs > 0 ) )
    {
        return -1;
    }

    mask.resize( P.rows(), P.cols() );
    norm = 0;

    for( int cc = 0; cc < P.cols(); ++cc )
    {
        for( int rr = 0; rr < P.rows(); ++rr )
        {
            if( std::fabs( P( rr, cc ) ) > thresh * maxAbs )
            {
                mask( rr, cc ) = 1;
                norm += static_cast<double>( P( rr, cc ) ) * P( rr, cc );
            }
            else
            {
                mask( rr, cc ) = 0;
            }
        }
    }

    if( !( norm > 0 ) )
    {
        return -1;
    }

    return 0;
}

/// Project a frame onto the reference pattern: sum_mask (im - base)*P / norm.
/**
 * \returns the projected response
 */
inline double projectResponse( const float *im,   /**< [in] the frame */
                               const float *base, /**< [in] the pre-poke frame */
                               const float *P,    /**< [in] the reference pattern */
                               const float *mask, /**< [in] the mask */
                               size_t       nPix, /**< [in] the number of pixels */
                               double       norm  /**< [in] the normalization, sum_mask P^2 */
)
{
    double sum = 0;

    for( size_t i = 0; i < nPix; ++i )
    {
        if( mask[i] != 0 )
        {
            sum += static_cast<double>( im[i] - base[i] ) * P[i];
        }
    }

    return sum / norm;
}

/// Find the first upward crossing of a level, by linear interpolation.
/**
 * \returns 0 on success
 * \returns -1 if the curve never crosses the level from below or the inputs are inconsistent
 */
inline int crossingTime( double                    &tcross, /**< [out] the crossing time */
                         const std::vector<double> &t,      /**< [in] the time axis */
                         const std::vector<double> &r,      /**< [in] the response */
                         double                     level   /**< [in] the level to cross */
)
{
    if( t.size() != r.size() || t.size() < 2 )
    {
        return -1;
    }

    for( size_t i = 1; i < r.size(); ++i )
    {
        if( r[i - 1] < level && r[i] >= level )
        {
            tcross = t[i - 1] + ( level - r[i - 1] ) * ( t[i] - t[i - 1] ) / ( r[i] - r[i - 1] );
            return 0;
        }
    }

    return -1;
}

/// Compute the response metrics for one delay.
/**
 * \returns 0 on success
 * \returns -1 if the curve does not cross 0.1, 0.5 and 0.9, or the inputs are inconsistent
 */
inline int computeMetrics( responseMetrics           &met,       /**< [out] the metrics */
                           const std::vector<double> &t,         /**< [in] time axis [us] relative to the command */
                           const std::vector<double> &rmean,     /**< [in] the mean response curve */
                           const std::vector<double> &rstd,      /**< [in] the trial-to-trial std of the response */
                           int                        nSettle,   /**< [in] trailing frames used for settleErr */
                           const std::vector<double> &delayErrs, /**< [in] achieved - requested delay per trial */
                           int                        nLate      /**< [in] the number of late trials */
)
{
    met = responseMetrics();

    if( t.size() != rmean.size() || t.size() != rstd.size() || rmean.size() < 2 )
    {
        return -1;
    }

    if( delayErrs.size() > 0 )
    {
        double sum = 0, sum2 = 0;
        for( double de : delayErrs )
        {
            sum += de;
            sum2 += de * de;
        }

        met.m_delayErrMean = sum / delayErrs.size();
        met.m_delayErrStd  = sqrt( std::max( 0.0, sum2 / delayErrs.size() - met.m_delayErrMean * met.m_delayErrMean ) );
        met.m_lateFrac     = static_cast<double>( nLate ) / delayErrs.size();
    }

    met.m_overshoot = *std::max_element( rmean.begin(), rmean.end() ) - 1.0;

    int    ns   = std::max( 1, std::min( nSettle, static_cast<int>( rmean.size() ) ) );
    double sse  = 0;
    for( size_t i = rmean.size() - ns; i < rmean.size(); ++i )
    {
        sse += ( rmean[i] - 1.0 ) * ( rmean[i] - 1.0 );
    }
    met.m_settleErr = sqrt( sse / ns );

    double t10, t50, t90;

    if( crossingTime( t50, t, rmean, 0.5 ) < 0 )
    {
        return -1;
    }

    met.m_t50 = t50;

    size_t jidx = 0;
    for( size_t i = 1; i < t.size(); ++i )
    {
        if( std::fabs( t[i] - t50 ) < std::fabs( t[jidx] - t50 ) )
        {
            jidx = i;
        }
    }

    met.m_jitter = rstd[jidx];

    if( crossingTime( t10, t, rmean, 0.1 ) < 0 || crossingTime( t90, t, rmean, 0.9 ) < 0 )
    {
        return -1;
    }

    met.m_rise = t90 - t10;

    return 0;
}

/// Average curves onto a common fine time grid by binning (the super-sampled response).
/** Bins of width dt start at the smallest time value.  Empty bins are NaN.
 *
 * \returns 0 on success
 * \returns -1 if dt <= 0 or the inputs are empty or inconsistent
 */
inline int resampleAverage( std::vector<double>                    &grid,   /**< [out] bin center times */
                            std::vector<double>                    &val,    /**< [out] bin averages */
                            const std::vector<std::vector<double>> &times,  /**< [in] time axis of each curve */
                            const std::vector<std::vector<double>> &curves, /**< [in] the curves */
                            double                                  dt      /**< [in] the bin width */
)
{
    if( !( dt > 0 ) || times.size() != curves.size() || times.size() == 0 )
    {
        return -1;
    }

    double tmin = std::numeric_limits<double>::max();
    double tmax = std::numeric_limits<double>::lowest();

    for( size_t c = 0; c < times.size(); ++c )
    {
        if( times[c].size() != curves[c].size() )
        {
            return -1;
        }

        for( size_t i = 0; i < times[c].size(); ++i )
        {
            if( !std::isfinite( times[c][i] ) || !std::isfinite( curves[c][i] ) )
            {
                continue;
            }

            tmin = std::min( tmin, times[c][i] );
            tmax = std::max( tmax, times[c][i] );
        }
    }

    if( tmax < tmin )
    {
        return -1;
    }

    size_t nBins = static_cast<size_t>( std::floor( ( tmax - tmin ) / dt ) ) + 1;

    std::vector<double> sums( nBins, 0.0 );
    std::vector<int>    counts( nBins, 0 );

    for( size_t c = 0; c < times.size(); ++c )
    {
        for( size_t i = 0; i < times[c].size(); ++i )
        {
            if( !std::isfinite( times[c][i] ) || !std::isfinite( curves[c][i] ) )
            {
                continue;
            }

            size_t b = static_cast<size_t>( std::floor( ( times[c][i] - tmin ) / dt ) );
            if( b >= nBins )
            {
                b = nBins - 1;
            }

            sums[b] += curves[c][i];
            ++counts[b];
        }
    }

    grid.resize( nBins );
    val.resize( nBins );

    for( size_t b = 0; b < nBins; ++b )
    {
        grid[b] = tmin + ( b + 0.5 ) * dt;
        val[b]  = ( counts[b] > 0 ) ? sums[b] / counts[b] : std::numeric_limits<double>::quiet_NaN();
    }

    return 0;
}

/// Select the best delay by the minimum of a metric.  Ties go to the lowest index (smallest delay).
/**
 * \returns 0 on success
 * \returns -1 if the criterion is unknown or no delay has a finite value of the metric
 */
inline int bestDelay( size_t                             &idx,      /**< [out] index of the best delay */
                      const std::vector<responseMetrics> &metrics,  /**< [in] the per-delay metrics */
                      const std::string                  &criterion /**< [in] "jitter", "rise", or "t50" */
)
{
    double responseMetrics::*field = nullptr;

    if( criterion == "jitter" )
    {
        field = &responseMetrics::m_jitter;
    }
    else if( criterion == "rise" )
    {
        field = &responseMetrics::m_rise;
    }
    else if( criterion == "t50" )
    {
        field = &responseMetrics::m_t50;
    }
    else
    {
        return -1;
    }

    bool   found = false;
    double best  = 0;

    for( size_t i = 0; i < metrics.size(); ++i )
    {
        double v = metrics[i].*field;

        if( !std::isfinite( v ) )
        {
            continue;
        }

        if( !found || v < best )
        {
            best  = v;
            idx   = i;
            found = true;
        }
    }

    return found ? 0 : -1;
}

/// Compute the SHA-256 digest of a byte string.
/**
 * \returns the digest as 64 lower-case hex characters
 */
inline std::string sha256Hex( const std::string &data /**< [in] the bytes to hash */ )
{
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };

    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };

    auto rotr = []( uint32_t x, int n ) { return ( x >> n ) | ( x << ( 32 - n ) ); };

    std::string msg = data;
    uint64_t    bitLen = static_cast<uint64_t>( data.size() ) * 8;

    msg.push_back( static_cast<char>( 0x80 ) );
    while( msg.size() % 64 != 56 )
    {
        msg.push_back( 0 );
    }
    for( int i = 7; i >= 0; --i )
    {
        msg.push_back( static_cast<char>( ( bitLen >> ( 8 * i ) ) & 0xff ) );
    }

    for( size_t chunk = 0; chunk < msg.size(); chunk += 64 )
    {
        uint32_t w[64];

        for( int i = 0; i < 16; ++i )
        {
            w[i] = ( static_cast<uint32_t>( static_cast<uint8_t>( msg[chunk + 4 * i] ) ) << 24 ) |
                   ( static_cast<uint32_t>( static_cast<uint8_t>( msg[chunk + 4 * i + 1] ) ) << 16 ) |
                   ( static_cast<uint32_t>( static_cast<uint8_t>( msg[chunk + 4 * i + 2] ) ) << 8 ) |
                   ( static_cast<uint32_t>( static_cast<uint8_t>( msg[chunk + 4 * i + 3] ) ) );
        }

        for( int i = 16; i < 64; ++i )
        {
            uint32_t s0 = rotr( w[i - 15], 7 ) ^ rotr( w[i - 15], 18 ) ^ ( w[i - 15] >> 3 );
            uint32_t s1 = rotr( w[i - 2], 17 ) ^ rotr( w[i - 2], 19 ) ^ ( w[i - 2] >> 10 );
            w[i]        = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];

        for( int i = 0; i < 64; ++i )
        {
            uint32_t S1    = rotr( e, 6 ) ^ rotr( e, 11 ) ^ rotr( e, 25 );
            uint32_t ch    = ( e & f ) ^ ( ~e & g );
            uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0    = rotr( a, 2 ) ^ rotr( a, 13 ) ^ rotr( a, 22 );
            uint32_t maj   = ( a & b ) ^ ( a & c ) ^ ( b & c );
            uint32_t temp2 = S0 + maj;

            hh = g;
            g  = f;
            f  = e;
            e  = d + temp1;
            d  = c;
            c  = b;
            b  = a;
            a  = temp1 + temp2;
        }

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    char out[65];
    for( int i = 0; i < 8; ++i )
    {
        snprintf( out + 8 * i, 9, "%08x", h[i] );
    }

    return std::string( out, 64 );
}

/// Compute the SHA-256 digest of a file.
/**
 * \returns 0 on success
 * \returns -1 if the file can not be read
 */
inline int sha256File( std::string       &hex, /**< [out] the digest as hex */
                       const std::string &path /**< [in] the file to hash */
)
{
    std::ifstream fin( path, std::ios::binary );

    if( !fin )
    {
        return -1;
    }

    std::stringstream ss;
    ss << fin.rdbuf();

    hex = sha256Hex( ss.str() );

    return 0;
}

/// Check the return value of an mxlib FITS write for success.
/** Works whether the mxlib version returns `mx::error_t` (noerror == 0) or `int` (0 on success).
 *
 * \returns true if \p rv is the zero-valued success code
 */
template <typename retT>
bool writeOk( const retT &rv /**< [in] the value returned by fitsFile::write */ )
{
    return rv == retT{};
}

/// Pack a timespec into integer nanoseconds.
/**
 * \returns the time in nanoseconds since the epoch
 */
inline int64_t tsToNs( const timespec &ts /**< [in] the time to pack */ )
{
    return static_cast<int64_t>( ts.tv_sec ) * 1000000000LL + ts.tv_nsec;
}

/// Unpack integer nanoseconds into a timespec.
/**
 * \returns the timespec
 */
inline timespec nsToTs( int64_t ns /**< [in] nanoseconds since the epoch */ )
{
    timespec ts;
    ts.tv_sec  = static_cast<time_t>( ns / 1000000000LL );
    ts.tv_nsec = static_cast<long>( ns % 1000000000LL );
    return ts;
}

} // namespace dmTemporalResponseMath

/// Tag type for the camWFS shmimMonitor parent.
/**
 * \ingroup dmTemporalResponse
 */
struct dmTemporalResponseWfsShmimT
{
    /// The configuration section, `wfscam`.
    static std::string configSection()
    {
        return "wfscam";
    };

    /// The INDI property prefix, `wfscam`.
    static std::string indiPrefix()
    {
        return "wfscam";
    };
};

/// The MagAO-X DM temporal response measurement application.
/** Triggers a DM poke (single actuator or FITS pattern) off the camWFS frame semaphore after a delay referenced to
 * the frame acquisition time, captures N frames per trial, and runs M/2 positive then M/2 negative trials per delay
 * on an evenly spaced delay grid.  Outputs averaged difference cubes and scalar response metrics.
 *
 * \ingroup dmTemporalResponse
 */
class dmTemporalResponse : public MagAOXApp<true>, public dev::shmimMonitor<dmTemporalResponse, dmTemporalResponseWfsShmimT>
{
    // Give the test harness access.
    friend class dmTemporalResponse_test;

    friend class dev::shmimMonitor<dmTemporalResponse, dmTemporalResponseWfsShmimT>;

  public:
    /// The camWFS shmimMonitor base type.
    typedef dev::shmimMonitor<dmTemporalResponse, dmTemporalResponseWfsShmimT> shmimMonitorT;

    /// The per-frame state machine states.
    enum class trialState : int
    {
        idle,      ///< No trial in progress; frames are ignored.
        armed,     ///< The next frame is the trigger frame.
        waitPoke,  ///< The poke deadline falls after a later frame; waiting for it.
        capturing  ///< The poke has been written; capturing N frames.
    };

    /// Result codes of a single trial.
    enum trialResult
    {
        resultStopped = -1, ///< Stop, shutdown, or a camera/fps change interrupted the trial.
        resultValid   = 0,  ///< The trial completed with contiguous frames.
        resultInvalid = 1,  ///< A frame counter gap invalidated the trial.
        resultTimeout = 2   ///< No trial completion within the timeout.
    };

    /// Snapshot of the parameters used for one run, taken at start so INDI changes can not affect a run.
    struct runParams
    {
        int m_dmIndex{ 0 }; ///< DM index NN.

        int m_dmChannel{ 7 }; ///< DM channel MM.

        dmTemporalResponseMath::pokeMode m_pokeMode{ dmTemporalResponseMath::pokeMode::actuator }; ///< The poke mode.

        std::vector<int> m_pokeX; ///< Actuator x (actuator mode).

        std::vector<int> m_pokeY; ///< Actuator y (actuator mode).

        std::string m_patternFile; ///< Pattern file path (pattern mode).

        float m_pokeAmp{ 0 }; ///< Poke amplitude.

        float m_maxCommand{ 1 }; ///< Command safety limit.

        int m_nDelays{ 10 }; ///< K.

        double m_delaySpan{ 0 }; ///< Configured span [us], <= 0 means one frame period.

        int m_nFrames{ 20 }; ///< N.

        int m_nTrials{ 20 }; ///< M.

        double m_settle{ 0.05 }; ///< Settle time [s].

        double m_trialTimeout{ 2.0 }; ///< Trial timeout [s].

        int m_maxRetries{ 5 }; ///< Max invalid-trial retries per delay.

        int m_nRef{ 10 }; ///< Reference-pass trial pairs.

        int m_nSettle{ 5 }; ///< Trailing frames used for the reference and settleErr.

        float m_maskThresh{ 0.1 }; ///< Mask threshold.

        int m_resampleFactor{ 10 }; ///< Super-sampling factor.

        std::string m_bestMetric{ "jitter" }; ///< Best-delay criterion.

        double m_maxLateFrac{ 0.1 }; ///< Late-fraction warning threshold.

        std::string m_baseDir{ "/home/xsup/dm_response" }; ///< Output root.

        double m_fps{ -1 }; ///< camWFS fps at run start.
    };

  protected:
    /** \name Configurable Parameters - Data
     *
     * @{
     */

    /// INDI device name of the WFS camera, used to read its fps.  Default is wfscam.shmimName.
    std::string m_wfsCamDevName;

    /// The DM index NN in dm<NN>disp<MM>: 0 woofer, 1 tweeter, 2 NCPC.
    int m_dmIndex{ 0 };

    /// The dmcomb channel MM in dm<NN>disp<MM>.
    int m_dmChannel{ 7 };

    /// The poke mode name, "actuator" or "pattern".
    std::string m_pokeModeName{ "actuator" };

    /// The x coordinate of the actuator to poke, exactly one entry required in actuator mode.
    std::vector<int> m_pokeX;

    /// The y coordinate of the actuator to poke, exactly one entry required in actuator mode.
    std::vector<int> m_pokeY;

    /// Path to a 2-D FITS DM pattern on this machine, used in pattern mode.
    std::string m_patternFile;

    /// The poke amplitude in DM command units.  The command is +/- amp (actuator) or +/- amp*pattern.
    float m_pokeAmp{ 0 };

    /// Maximum absolute DM command allowed in either mode.
    float m_maxCommand{ 1 };

    /// K, the number of evenly spaced delays.
    int m_nDelays{ 10 };

    /// Span of the delay grid in microseconds.  <= 0 means one camWFS frame period.
    double m_delaySpan{ 0 };

    /// N, the number of frames captured after each poke.
    int m_nFrames{ 20 };

    /// M, the total trials per delay (M/2 positive then M/2 negative), must be even.
    int m_nTrials{ 20 };

    /// Time in seconds to wait after zeroing the DM before arming the next trial.
    double m_settle{ 0.05 };

    /// Maximum time in seconds to wait for one trial to complete.
    double m_trialTimeout{ 2.0 };

    /// Maximum invalid-trial retries per delay before the run aborts.
    int m_maxRetries{ 5 };

    /// The number of +/- trial pairs in the reference pass.
    int m_nRef{ 10 };

    /// The number of trailing frames used for the steady-state reference and settleErr.
    int m_nSettle{ 5 };

    /// Mask threshold as a fraction of max|P|.
    float m_maskThresh{ 0.1 };

    /// Super-sampling factor: the combined response is binned at frame period / resampleFactor.
    int m_resampleFactor{ 10 };

    /// Criterion for choosing the best delay: jitter, rise, or t50.
    std::string m_bestMetric{ "jitter" };

    /// If the fraction of late pokes at a delay exceeds this, a warning is logged and the cube is flagged.
    double m_maxLateFrac{ 0.1 };

    /// Root of the output tree.  Each run creates a UTC-stamped sub-directory.
    std::string m_baseDir{ "/home/xsup/dm_response" };

    ///@}

    /** \name Camera State - Data
     *
     * @{
     */

    /// Width of the camWFS frames, set in allocate().
    uint32_t m_nx{ 0 };

    /// Height of the camWFS frames, set in allocate().
    uint32_t m_ny{ 0 };

    /// Pointer to a function to extract the camWFS pixel data as float.
    float ( *m_pixget )( void *, size_t ){ nullptr };

    /// The camWFS frame rate, from the camera's INDI fps property.  <= 0 if unknown.
    std::atomic<double> m_wfsFps{ -1 };

    /// Acquisition time [ns] of the most recent camWFS frame, used for the clock-domain check.  0 if none yet.
    std::atomic<int64_t> m_lastATimeNs{ 0 };

    ///@}

    /** \name DM State - Data
     *
     * @{
     */

    /// The DM channel stream written by this app.
    mx::improc::milkImage<float> m_dmStream;

    /// The resolved DM channel stream name for the current run.
    std::string m_dmStreamName;

    /// If not empty, used instead of dm<NN>disp<MM>.  Test harness use only.
    std::string m_dmStreamOverride;

    /// The pre-built positive poke command.
    mx::improc::eigenImage<float> m_cmdPos;

    /// The pre-built negative poke command.
    mx::improc::eigenImage<float> m_cmdNeg;

    /// An all-zero command of the DM size.
    mx::improc::eigenImage<float> m_cmdZero;

    /// The pattern loaded for the current run (pattern mode).
    mx::improc::eigenImage<float> m_pattern;

    /// SHA-256 of the pattern file for the current run.
    std::string m_patternSha;

    ///@}

    /** \name Trial State - Data
     * Shared between the shmimMonitor RT thread and the measurement thread.
     * @{
     */

    /// Guards the trial state handoff between the RT thread and the measurement thread.
    std::mutex m_trialMutex;

    /// The per-frame state machine state.
    std::atomic<trialState> m_trialState{ trialState::idle };

    /// The requested delay [us] of the current trial.
    double m_curDelay{ 0 };

    /// The poke sign of the current trial, +1 or -1.
    int m_curSign{ 1 };

    /// The number of frames to capture per trial for the current run.
    int m_curNFrames{ 0 };

    /// The camWFS frame period [us] for the current run, <= 0 if unknown.
    double m_framePeriodUs{ -1 };

    /// Acquisition time of the trigger frame.
    timespec m_trigATime{ 0, 0 };

    /// The poke deadline, trigger atime + delay.
    timespec m_tPoke{ 0, 0 };

    /// The time the poke command was written.
    timespec m_tCmd{ 0, 0 };

    /// Whether the current trial is still valid (no frame gap).
    bool m_trialValid{ true };

    /// Whether the poke deadline had already passed when checked.
    bool m_trialLate{ false };

    /// The frame counter of the last frame seen in this trial.
    uint64_t m_trialCnt0{ 0 };

    /// The number of frames captured so far in this trial.
    int m_nCaptured{ 0 };

    /// The trial buffer [nx, ny, N+1]: plane 0 is the last pre-poke frame, planes 1..N are post-poke frames.
    mx::improc::eigenCube<float> m_trialBuf;

    /// Acquisition times of the frames in m_trialBuf.
    std::vector<timespec> m_trialTimes;

    /// Posted by the RT thread when a trial completes.
    sem_t m_trialSem;

    /// Whether the semaphores were initialized in the constructor.
    bool m_semsInit{ false };

    /// The clock used for the poke busy-wait and command timestamp.  Injectable for testing.
    timespec ( *m_clock )(){ &dmTemporalResponseMath::realtimeNow };

    ///@}

    /** \name Measurement Thread - Data
     *
     * @{
     */

    /// Guards the configurable parameters against changes while a run is starting.
    std::mutex m_paramMutex;

    /// The parameters for the current run.
    runParams m_run;

    /// True while a run is in progress.
    std::atomic<bool> m_running{ false };

    /// Set to request that the current run stop.
    std::atomic<bool> m_stopRequested{ false };

    /// Set if the camWFS fps changes during a run.
    std::atomic<bool> m_fpsChanged{ false };

    /// Set if the camWFS stream is re-allocated during a run.
    std::atomic<bool> m_camChanged{ false };

    /// Posted to start a run.
    sem_t m_startSem;

    /// Priority of the measurement thread.  Normal (0), since timing-critical work is on the shmimMonitor thread.
    int m_measThreadPrio{ 0 };

    /// The measurement thread.
    std::thread m_measThread;

    /// Synchronizer to ensure the measurement thread initializes before doing dangerous things.
    bool m_measThreadInit{ true };

    /// Measurement thread PID.
    pid_t m_measThreadID{ 0 };

    /// The property to hold the measurement thread details.
    pcf::IndiProperty m_measThreadProp;

    ///@}

    /** \name Results - Data
     *
     * @{
     */

    /// The delay grid of the current or last run [us].
    std::vector<double> m_delays;

    /// The resolved span of the delay grid [us].
    double m_span{ 0 };

    /// Guards the result strings and values read by appLogic for INDI.
    std::mutex m_resultsMutex;

    /// The delay grid as comma-separated text, for INDI.
    std::string m_delaysText;

    /// Per-delay t50 values as comma-separated text, for INDI.
    std::string m_t50Text;

    /// Per-delay rise times as comma-separated text, for INDI.
    std::string m_riseText;

    /// Per-delay jitter values as comma-separated text, for INDI.
    std::string m_jitterText;

    /// The best delay and its t50, rise, and jitter, for INDI.
    std::vector<double> m_bestValues{ -1, -1, -1, -1 };

    /// Per-delay metrics of the current or last run.
    std::vector<dmTemporalResponseMath::responseMetrics> m_metrics;

    /// Per-delay mean response curves r̄_d(k).
    std::vector<std::vector<double>> m_respMean;

    /// Per-delay std of the response curves.
    std::vector<std::vector<double>> m_respStd;

    /// Per-delay mean time axes [us] relative to the command.
    std::vector<std::vector<double>> m_respTime;

    /// The reference pattern P.
    mx::improc::eigenImage<float> m_refP;

    /// The reference mask.
    mx::improc::eigenImage<float> m_refMask;

    /// The projection normalization, sum_mask P^2.
    double m_refNorm{ 0 };

    /// Index of the best delay, if found.
    size_t m_bestIdx{ 0 };

    /// Whether m_bestIdx is valid.
    bool m_haveBest{ false };

    /// The directory of the current or last run.
    std::string m_runDir;

    /// The run status: idle, running, done, error, or stopped.
    std::string m_runStatus{ "idle" };

    /// The run phase: none, reference, measuring, or analyzing.
    std::string m_runPhase{ "none" };

    /// Index of the delay in progress.
    std::atomic<int> m_progDelayIdx{ 0 };

    /// Delay in progress [us].
    std::atomic<double> m_progDelayUs{ 0 };

    /// Sign in progress.
    std::atomic<int> m_progSign{ 1 };

    /// Trial number in progress.
    std::atomic<int> m_progTrial{ 0 };

    /// Invalid trials at the current delay.
    std::atomic<int> m_progInvalid{ 0 };

    /// Late-poke fraction at the last completed delay.
    std::atomic<double> m_progLateFrac{ 0 };

    /// Description of the loaded pattern, or the reason it failed validation.
    std::string m_patternInfo;

    /// Live shmim of the reference pattern.
    mx::improc::milkImage<float> m_refStream;

    /// Live shmim of the mean response curves, N x K.
    mx::improc::milkImage<float> m_respStream;

    /// Live shmim of the super-sampled response.
    mx::improc::milkImage<float> m_respAvgStream;

    ///@}

  public:
    /// Default c'tor.
    dmTemporalResponse();

    /// D'tor, declared and defined for noexcept.
    ~dmTemporalResponse() noexcept;

    /** \name MagAOXApp Interface
     *
     * @{
     */

    /// Set up the application configuration.
    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] configuration from which to load */ );

    /// Load the application configuration.
    virtual void loadConfig();

    /// Perform application startup: INDI properties, semaphores, and threads.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int appStartup();

    /// Implementation of the FSM for dmTemporalResponse.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shut the application down, stopping any run and zeroing the DM channel.
    /**
     * \returns 0 on success
     */
    virtual int appShutdown();

    /// Create and register all INDI properties.  Separated from appStartup so tests can exercise the callbacks.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int createIndiProperties();

    ///@}

    /** \name Configurable Parameters
     *
     * @{
     */

    /// Get the WFS camera INDI device name.
    const std::string &wfsCamDevName() const;

    /// Get the DM index NN.
    int dmIndex() const;

    /// Get the DM channel MM.
    int dmChannel() const;

    /// Get the poke mode name.
    const std::string &pokeModeName() const;

    /// Get the actuator x coordinates.
    const std::vector<int> &pokeX() const;

    /// Get the actuator y coordinates.
    const std::vector<int> &pokeY() const;

    /// Get the pattern file path.
    const std::string &patternFile() const;

    /// Get the poke amplitude.
    float pokeAmp() const;

    /// Get the command safety limit.
    float maxCommand() const;

    /// Get K, the number of delays.
    int nDelays() const;

    /// Get the delay span [us].
    double delaySpan() const;

    /// Get N, the frames per trial.
    int nFrames() const;

    /// Get M, the trials per delay.
    int nTrials() const;

    /// Get the settle time [s].
    double settle() const;

    /// Get the trial timeout [s].
    double trialTimeout() const;

    /// Get the maximum retries per delay.
    int maxRetries() const;

    /// Get the reference-pass trial pairs.
    int nRef() const;

    /// Get the trailing settle frames.
    int nSettle() const;

    /// Get the mask threshold.
    float maskThresh() const;

    /// Get the super-sampling factor.
    int resampleFactor() const;

    /// Get the best-delay criterion.
    const std::string &bestMetric() const;

    /// Get the late-fraction warning threshold.
    double maxLateFrac() const;

    /// Get the output root directory.
    const std::string &baseDir() const;

    ///@}

    /** \name shmimMonitor Interface
     *
     * @{
     */

    /// Allocate for a new camWFS stream: record its size and pixel accessor.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int allocate( const dmTemporalResponseWfsShmimT &dummy /**< [in] tag to differentiate shmimMonitor parents */ );

    /// Process a camWFS frame: read its metadata and run the per-frame state machine.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int processImage( void                              *curr_src, /**< [in] pointer to the start of the frame */
                      const dmTemporalResponseWfsShmimT &dummy     /**< [in] tag to differentiate parents */
    );

    ///@}

    /** \name Measurement Logic
     *
     * @{
     */

    /// The per-frame state machine: trigger, delay busy-wait, poke, and capture.
    /** Called on the shmimMonitor RT thread for every frame.  Only copies data; no arithmetic beyond timing.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int processFrame( void           *src,   /**< [in] pointer to the frame data */
                      const timespec &atime, /**< [in] the frame acquisition time */
                      uint64_t        cnt0   /**< [in] the frame counter */
    );

    /// Copy a camWFS frame into a plane of the trial buffer as float, and record its acquisition time.
    void copyFrame( void           *src,   /**< [in] pointer to the frame data */
                    int             plane, /**< [in] the trial buffer plane, 0 for the baseline */
                    const timespec &atime  /**< [in] the frame acquisition time */
    );

    /// Write a command to the DM channel.  Virtual so the test harness can observe DM writes.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    virtual int writeDM( const mx::improc::eigenImage<float> &cmd /**< [in] the command to write */ );

    /// Zero the DM channel, if it is open.
    void zeroDM();

    /// Snapshot the parameters for a run from the configurable members.
    void snapshotParams( runParams &params /**< [out] the snapshot */ ) const;

    /// Validate the run parameters, open the DM, build the commands, and create the output directory.
    /**
     * \returns 0 on success
     * \returns -1 on a validation or setup failure, which is logged; the DM is not written
     */
    int prepareRun( const timespec &runStart /**< [in] the run start time, names the output directory */ );

    /// Run one trial at a delay and sign.
    /**
     * \returns a trialResult code
     */
    int runTrial( double delay, /**< [in] the delay [us] */
                  int    sign   /**< [in] +1 or -1 */
    );

    /// Run a +/- trial set at one delay, accumulating the image sums and the per-trial response curves.
    /** When \p project is false (the reference pass) no response curves are computed.
     *
     * \returns 0 on success
     * \returns -1 on abort (retries exceeded, timeout, stop, or camera change)
     */
    int runTrialSet( double                             delay,     /**< [in] the delay [us] */
                     int                                nPerSign,  /**< [in] valid trials per sign */
                     bool                               project,   /**< [in] compute response curves if true */
                     mx::improc::eigenCube<double>     &sumPos,    /**< [out] sum of positive trials */
                     mx::improc::eigenCube<double>     &sumNeg,    /**< [out] sum of negative trials */
                     std::vector<std::vector<double>>  &curves,    /**< [out] per-trial response curves */
                     std::vector<std::vector<double>>  &times,     /**< [out] per-trial time axes [us] */
                     std::vector<double>               &delayErrs, /**< [out] per-trial delay errors [us] */
                     int                               &nLate      /**< [out] the number of late trials */
    );

    /// Run the reference pass and build P, the mask, and the normalization.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int runReference();

    /// Execute one complete run: prepare, reference pass, the delay grid, analysis, and outputs.
    /** This is the measurement thread body for one run, and can be called directly by tests.
     *
     * \returns 0 on success
     * \returns -1 on error or stop
     */
    int runMeasurement();

    /// Request that a run start.
    /**
     * \returns 0 on success
     * \returns -1 if a run is already in progress
     */
    int requestStart();

    /// Request that the current run stop.
    void requestStop();

    /// Set the run status and phase reported over INDI.
    void setRunState( const std::string &status, /**< [in] idle, running, done, error, or stopped */
                      const std::string &phase   /**< [in] none, reference, measuring, or analyzing */
    );

    /// Set the pattern description reported over INDI.
    void setPatternInfo( const std::string &info /**< [in] the description or validation error */ );

    ///@}

    /** \name Output
     *
     * @{
     */

    /// Write the averaged cube for one delay.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int writeCube( size_t                              idx,       /**< [in] the delay index */
                   const mx::improc::eigenCube<float> &cube,      /**< [in] the averaged cube */
                   const std::vector<double>          &delayErrs, /**< [in] per-trial delay errors [us] */
                   int                                 nInvalid   /**< [in] the number of invalid trials */
    );

    /// Write reference.fits: P and the mask.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int writeReference();

    /// Write the summary files: curves, metrics, and the super-sampled response.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int writeSummary( const std::vector<double> &grid, /**< [in] the super-sampled time grid [us] */
                      const std::vector<double> &val   /**< [in] the super-sampled response */
    );

    /// Append the common run header keywords.
    void appendRunHeader( mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> &fh /**< [in,out] the header */ );

    ///@}

    /** \name Measurement Thread
     *
     * @{
     */

    /// Thread starter, called by threadStart on thread construction.  Calls measThreadExec.
    static void measThreadStart( dmTemporalResponse *s /**< [in] a pointer to a dmTemporalResponse instance */ );

    /// Execute the measurement thread main loop: wait for a start request and run.
    void measThreadExec();

    ///@}

    /** \name INDI Interface
     *
     * @{
     */

  protected:
    /// Apply a new target for a tunable, rejected while a run is in progress.
    /**
     * \returns 0 on success
     * \returns -1 on error or if a run is in progress
     */
    template <typename T>
    int tunableCallback( pcf::IndiProperty       &local,  /**< [in,out] the local property */
                         T                       &member, /**< [out] the member to update */
                         const pcf::IndiProperty &ipRecv  /**< [in] the received property */
    );

    pcf::IndiProperty m_indiP_dmIndex; ///< DM index NN.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_dmIndex );

    pcf::IndiProperty m_indiP_dmChannel; ///< DM channel MM.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_dmChannel );

    pcf::IndiProperty m_indiP_pokeMode; ///< Poke mode selection: actuator or pattern.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_pokeMode );

    pcf::IndiProperty m_indiP_pokeX; ///< Actuator x.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_pokeX );

    pcf::IndiProperty m_indiP_pokeY; ///< Actuator y.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_pokeY );

    pcf::IndiProperty m_indiP_patternFile; ///< Pattern file path.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_patternFile );

    pcf::IndiProperty m_indiP_pokeAmp; ///< Poke amplitude.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_pokeAmp );

    pcf::IndiProperty m_indiP_nDelays; ///< K.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_nDelays );

    pcf::IndiProperty m_indiP_delaySpan; ///< Delay span [us].
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_delaySpan );

    pcf::IndiProperty m_indiP_nFrames; ///< N.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_nFrames );

    pcf::IndiProperty m_indiP_nTrials; ///< M.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_nTrials );

    pcf::IndiProperty m_indiP_settle; ///< Settle time [s].
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_settle );

    pcf::IndiProperty m_indiP_nSettle; ///< Trailing settle frames.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_nSettle );

    pcf::IndiProperty m_indiP_maskThresh; ///< Mask threshold.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_maskThresh );

    pcf::IndiProperty m_indiP_bestMetric; ///< Best-delay criterion.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_bestMetric );

    pcf::IndiProperty m_indiP_start; ///< Request switch to start a run.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_start );

    pcf::IndiProperty m_indiP_stop; ///< Request switch to stop a run.
    INDI_NEWCALLBACK_DECL( dmTemporalResponse, m_indiP_stop );

    pcf::IndiProperty m_indiP_wfsFps; ///< The WFS camera fps, a set property.
    INDI_SETCALLBACK_DECL( dmTemporalResponse, m_indiP_wfsFps );

    pcf::IndiProperty m_indiP_dmStream; ///< RO: the resolved DM stream name.

    pcf::IndiProperty m_indiP_runState; ///< RO: run status and phase.

    pcf::IndiProperty m_indiP_progress; ///< RO: delay index, delay, sign, trial, invalid count, late fraction.

    pcf::IndiProperty m_indiP_delays; ///< RO: the delay grid, comma separated.

    pcf::IndiProperty m_indiP_results; ///< RO: per-delay t50, rise, and jitter, comma separated.

    pcf::IndiProperty m_indiP_best; ///< RO: the best delay and its metrics.

    pcf::IndiProperty m_indiP_patternInfo; ///< RO: the loaded pattern description or validation error.

    pcf::IndiProperty m_indiP_output; ///< RO: the run output directory.

    ///@}
};

inline dmTemporalResponse::dmTemporalResponse() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    // Initialized here, not in appStartup, so runMeasurement() can be exercised directly.
    m_semsInit = ( sem_init( &m_trialSem, 0, 0 ) == 0 && sem_init( &m_startSem, 0, 0 ) == 0 );

    return;
}

inline dmTemporalResponse::~dmTemporalResponse() noexcept
{
    if( m_semsInit )
    {
        sem_destroy( &m_trialSem );
        sem_destroy( &m_startSem );
    }
}

inline void dmTemporalResponse::setupConfig()
{
    SHMIMMONITORT_SETUP_CONFIG( shmimMonitorT, config );

    // Default camera stream, can be overridden by wfscam.shmimName
    shmimMonitorT::m_shmimName = "camwfs";

    config.add( "wfscam.camDevName",
                "",
                "wfscam.camDevName",
                argType::Required,
                "wfscam",
                "camDevName",
                false,
                "string",
                "INDI device name of the WFS camera.  Default is wfscam.shmimName." );

    config.add( "dm.index",
                "",
                "dm.index",
                argType::Required,
                "dm",
                "index",
                false,
                "int",
                "DM index NN in dm<NN>disp<MM>: 0 woofer, 1 tweeter, 2 NCPC.  Default 0." );

    config.add( "dm.channel",
                "",
                "dm.channel",
                argType::Required,
                "dm",
                "channel",
                false,
                "int",
                "dmcomb channel MM in dm<NN>disp<MM>.  Default 7." );

    config.add( "poke.mode",
                "",
                "poke.mode",
                argType::Required,
                "poke",
                "mode",
                false,
                "string",
                "Poke mode: actuator or pattern.  Default actuator." );

    config.add( "poke.x",
                "",
                "poke.x",
                argType::Required,
                "poke",
                "x",
                false,
                "vector<int>",
                "x coordinate of the actuator to poke (exactly one entry)." );

    config.add( "poke.y",
                "",
                "poke.y",
                argType::Required,
                "poke",
                "y",
                false,
                "vector<int>",
                "y coordinate of the actuator to poke (exactly one entry)." );

    config.add( "poke.patternFile",
                "",
                "poke.patternFile",
                argType::Required,
                "poke",
                "patternFile",
                false,
                "string",
                "Path to a 2-D FITS DM pattern, used in pattern mode." );

    config.add( "poke.amp",
                "",
                "poke.amp",
                argType::Required,
                "poke",
                "amp",
                false,
                "float",
                "Poke amplitude in DM command units.  Must be non-zero to start.  Default 0." );

    config.add( "poke.maxCommand",
                "",
                "poke.maxCommand",
                argType::Required,
                "poke",
                "maxCommand",
                false,
                "float",
                "Maximum absolute DM command allowed in either mode.  Default 1." );

    config.add( "poke.nDelays",
                "",
                "poke.nDelays",
                argType::Required,
                "poke",
                "nDelays",
                false,
                "int",
                "K, the number of evenly spaced delays.  Default 10." );

    config.add( "poke.delaySpan",
                "",
                "poke.delaySpan",
                argType::Required,
                "poke",
                "delaySpan",
                false,
                "float",
                "Span of the delay grid in us.  <= 0 means one camWFS frame period.  Default 0." );

    config.add( "poke.nFrames",
                "",
                "poke.nFrames",
                argType::Required,
                "poke",
                "nFrames",
                false,
                "int",
                "N, the number of frames captured per trial.  Default 20." );

    config.add( "poke.nTrials",
                "",
                "poke.nTrials",
                argType::Required,
                "poke",
                "nTrials",
                false,
                "int",
                "M, total trials per delay (M/2 positive then M/2 negative), must be even.  Default 20." );

    config.add( "poke.settle",
                "",
                "poke.settle",
                argType::Required,
                "poke",
                "settle",
                false,
                "float",
                "Seconds to wait after zeroing the DM before the next trial.  Default 0.05." );

    config.add( "poke.trialTimeout",
                "",
                "poke.trialTimeout",
                argType::Required,
                "poke",
                "trialTimeout",
                false,
                "float",
                "Maximum seconds to wait for one trial.  Default 2." );

    config.add( "poke.maxRetries",
                "",
                "poke.maxRetries",
                argType::Required,
                "poke",
                "maxRetries",
                false,
                "int",
                "Maximum invalid-trial retries per delay before aborting.  Default 5." );

    config.add( "analysis.nRef",
                "",
                "analysis.nRef",
                argType::Required,
                "analysis",
                "nRef",
                false,
                "int",
                "Number of +/- trial pairs in the reference pass.  Default 10." );

    config.add( "analysis.nSettle",
                "",
                "analysis.nSettle",
                argType::Required,
                "analysis",
                "nSettle",
                false,
                "int",
                "Trailing frames used for the reference and settleErr.  Default 5." );

    config.add( "analysis.maskThresh",
                "",
                "analysis.maskThresh",
                argType::Required,
                "analysis",
                "maskThresh",
                false,
                "float",
                "Mask threshold as a fraction of max|P|.  Default 0.1." );

    config.add( "analysis.resampleFactor",
                "",
                "analysis.resampleFactor",
                argType::Required,
                "analysis",
                "resampleFactor",
                false,
                "int",
                "Super-sampling factor for the combined response.  Default 10." );

    config.add( "analysis.bestMetric",
                "",
                "analysis.bestMetric",
                argType::Required,
                "analysis",
                "bestMetric",
                false,
                "string",
                "Best-delay criterion: jitter, rise, or t50.  Default jitter." );

    config.add( "analysis.maxLateFrac",
                "",
                "analysis.maxLateFrac",
                argType::Required,
                "analysis",
                "maxLateFrac",
                false,
                "float",
                "Late-poke fraction above which a warning is logged and the cube flagged.  Default 0.1." );

    config.add( "output.baseDir",
                "",
                "output.baseDir",
                argType::Required,
                "output",
                "baseDir",
                false,
                "string",
                "Root of the output tree.  Default /home/xsup/dm_response." );
}

inline int dmTemporalResponse::loadConfigImpl( mx::app::appConfigurator &_config )
{
    SHMIMMONITORT_LOAD_CONFIG( shmimMonitorT, _config );

    m_wfsCamDevName = shmimMonitorT::m_shmimName;
    _config( m_wfsCamDevName, "wfscam.camDevName" );

    _config( m_dmIndex, "dm.index" );
    _config( m_dmChannel, "dm.channel" );
    _config( m_pokeModeName, "poke.mode" );
    _config( m_pokeX, "poke.x" );
    _config( m_pokeY, "poke.y" );
    _config( m_patternFile, "poke.patternFile" );
    _config( m_pokeAmp, "poke.amp" );
    _config( m_maxCommand, "poke.maxCommand" );
    _config( m_nDelays, "poke.nDelays" );
    _config( m_delaySpan, "poke.delaySpan" );
    _config( m_nFrames, "poke.nFrames" );
    _config( m_nTrials, "poke.nTrials" );
    _config( m_settle, "poke.settle" );
    _config( m_trialTimeout, "poke.trialTimeout" );
    _config( m_maxRetries, "poke.maxRetries" );
    _config( m_nRef, "analysis.nRef" );
    _config( m_nSettle, "analysis.nSettle" );
    _config( m_maskThresh, "analysis.maskThresh" );
    _config( m_resampleFactor, "analysis.resampleFactor" );
    _config( m_bestMetric, "analysis.bestMetric" );
    _config( m_maxLateFrac, "analysis.maxLateFrac" );
    _config( m_baseDir, "output.baseDir" );

    // Parameters are validated at run start so that INDI changes are validated the same way.
    dmTemporalResponseMath::pokeMode mode;
    if( dmTemporalResponseMath::parsePokeMode( mode, m_pokeModeName ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid poke.mode: " + m_pokeModeName } );
    }

    return 0;
}

inline void dmTemporalResponse::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = 1;
    }
}

inline int dmTemporalResponse::createIndiProperties()
{
    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_dmIndex, "dm_index", 0, 2, 1, "%d", "DM Index (NN)", "DM" );
    m_indiP_dmIndex["current"] = m_dmIndex;
    m_indiP_dmIndex["target"]  = m_dmIndex;

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_dmChannel, "dm_channel", 0, 99, 1, "%d", "DM Channel (MM)", "DM" );
    m_indiP_dmChannel["current"] = m_dmChannel;
    m_indiP_dmChannel["target"]  = m_dmChannel;

    if( createStandardIndiSelectionSw( m_indiP_pokeMode, "poke_mode", { "actuator", "pattern" }, "Poke Mode", "Poke" ) <
        0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from createStandardIndiSelectionSw" } );
    }
    m_indiP_pokeMode[m_pokeModeName].setSwitchState( pcf::IndiElement::On );
    if( registerIndiPropertyNew( m_indiP_pokeMode, INDI_NEWCALLBACK( m_indiP_pokeMode ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error from registerIndiPropertyNew" } );
    }

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_pokeX, "poke_x", 0, 2047, 1, "%d", "Actuator x", "Poke" );
    m_indiP_pokeX["current"] = ( m_pokeX.size() > 0 ) ? m_pokeX[0] : -1;
    m_indiP_pokeX["target"]  = ( m_pokeX.size() > 0 ) ? m_pokeX[0] : -1;

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_pokeY, "poke_y", 0, 2047, 1, "%d", "Actuator y", "Poke" );
    m_indiP_pokeY["current"] = ( m_pokeY.size() > 0 ) ? m_pokeY[0] : -1;
    m_indiP_pokeY["target"]  = ( m_pokeY.size() > 0 ) ? m_pokeY[0] : -1;

    CREATE_REG_INDI_NEW_TEXT( m_indiP_patternFile, "pattern_file", "Pattern File", "Poke" );
    m_indiP_patternFile["current"] = m_patternFile;
    m_indiP_patternFile["target"]  = m_patternFile;

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_pokeAmp, "poke_amp", -1, 1, 1e-3, "%0.3f", "Poke Amplitude", "Poke" );
    m_indiP_pokeAmp["current"] = m_pokeAmp;
    m_indiP_pokeAmp["target"]  = m_pokeAmp;

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_nDelays, "nDelays", 1, 1000, 1, "%d", "Number of Delays", "Delays" );
    m_indiP_nDelays["current"] = m_nDelays;
    m_indiP_nDelays["target"]  = m_nDelays;

    CREATE_REG_INDI_NEW_NUMBERD( m_indiP_delaySpan, "delaySpan", 0, 1e7, 1, "%0.1f", "Delay Span [us]", "Delays" );
    m_indiP_delaySpan["current"] = m_delaySpan;
    m_indiP_delaySpan["target"]  = m_delaySpan;

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_nFrames, "nFrames", 1, 10000, 1, "%d", "Frames per Trial", "Trials" );
    m_indiP_nFrames["current"] = m_nFrames;
    m_indiP_nFrames["target"]  = m_nFrames;

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_nTrials, "nTrials", 2, 10000, 2, "%d", "Trials per Delay", "Trials" );
    m_indiP_nTrials["current"] = m_nTrials;
    m_indiP_nTrials["target"]  = m_nTrials;

    CREATE_REG_INDI_NEW_NUMBERD( m_indiP_settle, "settle", 0, 60, 0.01, "%0.3f", "Settle Time [s]", "Trials" );
    m_indiP_settle["current"] = m_settle;
    m_indiP_settle["target"]  = m_settle;

    CREATE_REG_INDI_NEW_NUMBERI( m_indiP_nSettle, "nSettle", 1, 10000, 1, "%d", "Settle Frames", "Analysis" );
    m_indiP_nSettle["current"] = m_nSettle;
    m_indiP_nSettle["target"]  = m_nSettle;

    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_maskThresh, "maskThresh", 0, 1, 0.01, "%0.2f", "Mask Threshold", "Analysis" );
    m_indiP_maskThresh["current"] = m_maskThresh;
    m_indiP_maskThresh["target"]  = m_maskThresh;

    CREATE_REG_INDI_NEW_TEXT( m_indiP_bestMetric, "bestMetric", "Best-Delay Metric", "Analysis" );
    m_indiP_bestMetric["current"] = m_bestMetric;
    m_indiP_bestMetric["target"]  = m_bestMetric;

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_start, "start" );

    CREATE_REG_INDI_NEW_REQUESTSWITCH( m_indiP_stop, "stop" );

    REG_INDI_SETPROP( m_indiP_wfsFps, m_wfsCamDevName, std::string( "fps" ) );

    createROIndiText( m_indiP_dmStream, "dm_stream", "name", "DM Stream", "DM" );
    std::string sname;
    dmTemporalResponseMath::dmStreamName( sname, m_dmIndex, m_dmChannel );
    m_indiP_dmStream["name"] = sname;
    registerIndiPropertyReadOnly( m_indiP_dmStream );

    createROIndiText( m_indiP_runState, "run_state", "status", "Run State", "Run" );
    m_indiP_runState.add( pcf::IndiElement( "phase" ) );
    m_indiP_runState["status"] = m_runStatus;
    m_indiP_runState["phase"]  = m_runPhase;
    registerIndiPropertyReadOnly( m_indiP_runState );

    createROIndiNumber( m_indiP_progress, "progress", "Progress", "Run" );
    indi::addNumberElement<double>( m_indiP_progress, "delay_index", 0, 1e6, 1, "%0.0f", "Delay Index" );
    indi::addNumberElement<double>( m_indiP_progress, "delay_us", 0, 1e9, 1, "%0.1f", "Delay [us]" );
    indi::addNumberElement<double>( m_indiP_progress, "sign", -1, 1, 1, "%0.0f", "Sign" );
    indi::addNumberElement<double>( m_indiP_progress, "trial", 0, 1e6, 1, "%0.0f", "Trial" );
    indi::addNumberElement<double>( m_indiP_progress, "n_invalid", 0, 1e6, 1, "%0.0f", "Invalid Trials" );
    indi::addNumberElement<double>( m_indiP_progress, "late_frac", 0, 1, 0.01, "%0.2f", "Late Fraction" );
    registerIndiPropertyReadOnly( m_indiP_progress );

    createROIndiText( m_indiP_delays, "delays", "values", "Delays [us]", "Results" );
    registerIndiPropertyReadOnly( m_indiP_delays );

    createROIndiText( m_indiP_results, "results", "t50", "Results", "Results" );
    m_indiP_results.add( pcf::IndiElement( "rise" ) );
    m_indiP_results.add( pcf::IndiElement( "jitter" ) );
    registerIndiPropertyReadOnly( m_indiP_results );

    createROIndiNumber( m_indiP_best, "best", "Best Delay", "Results" );
    indi::addNumberElement<double>( m_indiP_best, "delay_us", -1e9, 1e9, 1, "%0.1f", "Delay [us]" );
    indi::addNumberElement<double>( m_indiP_best, "t50", -1e9, 1e9, 1, "%0.1f", "t50 [us]" );
    indi::addNumberElement<double>( m_indiP_best, "rise", -1e9, 1e9, 1, "%0.1f", "Rise [us]" );
    indi::addNumberElement<double>( m_indiP_best, "jitter", -1e9, 1e9, 1e-4, "%0.4f", "Jitter" );
    registerIndiPropertyReadOnly( m_indiP_best );

    createROIndiText( m_indiP_patternInfo, "pattern_info", "info", "Pattern Info", "Poke" );
    registerIndiPropertyReadOnly( m_indiP_patternInfo );

    createROIndiText( m_indiP_output, "output", "dir", "Output Directory", "Run" );
    registerIndiPropertyReadOnly( m_indiP_output );

    return 0;
}

inline int dmTemporalResponse::appStartup()
{
    if( createIndiProperties() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error creating INDI properties" } );
    }

    if( !m_semsInit )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "semaphore initialization failed" } );
    }

    SHMIMMONITORT_APP_STARTUP( shmimMonitorT );

    if( threadStart( m_measThread,
                     m_measThreadInit,
                     m_measThreadID,
                     m_measThreadProp,
                     m_measThreadPrio,
                     "",
                     "measurement",
                     this,
                     measThreadStart ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    // The shmimMonitor runs while OPERATING.  Run status is reported via run_state.
    state( stateCodes::OPERATING );

    return 0;
}

inline int dmTemporalResponse::appLogic()
{
    SHMIMMONITORT_APP_LOGIC( shmimMonitorT );

    // Check that the measurement thread is still alive.
    try
    {
        if( pthread_tryjoin_np( m_measThread.native_handle(), 0 ) == 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "measurement thread has exited" } );
        }
    }
    catch( ... )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "measurement thread has exited" } );
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITORT_UPDATE_INDI( shmimMonitorT );

    // Snapshot the tunables under the lock held by the INDI callbacks
    runParams cur;
    { //mutex scope
        std::lock_guard<std::mutex> plock( m_paramMutex );
        snapshotParams( cur );
    }

    updateIfChanged( m_indiP_dmIndex, "current", cur.m_dmIndex );
    updateIfChanged( m_indiP_dmChannel, "current", cur.m_dmChannel );
    updateIfChanged( m_indiP_pokeX, "current", ( cur.m_pokeX.size() > 0 ) ? cur.m_pokeX[0] : -1 );
    updateIfChanged( m_indiP_pokeY, "current", ( cur.m_pokeY.size() > 0 ) ? cur.m_pokeY[0] : -1 );
    updateIfChanged( m_indiP_patternFile, "current", cur.m_patternFile );
    updateIfChanged( m_indiP_pokeAmp, "current", cur.m_pokeAmp );
    updateIfChanged( m_indiP_nDelays, "current", cur.m_nDelays );
    updateIfChanged( m_indiP_delaySpan, "current", cur.m_delaySpan );
    updateIfChanged( m_indiP_nFrames, "current", cur.m_nFrames );
    updateIfChanged( m_indiP_nTrials, "current", cur.m_nTrials );
    updateIfChanged( m_indiP_settle, "current", cur.m_settle );
    updateIfChanged( m_indiP_nSettle, "current", cur.m_nSettle );
    updateIfChanged( m_indiP_maskThresh, "current", cur.m_maskThresh );
    updateIfChanged( m_indiP_bestMetric, "current", cur.m_bestMetric );

    std::string sname;
    dmTemporalResponseMath::dmStreamName( sname, cur.m_dmIndex, cur.m_dmChannel );
    if( m_dmStreamOverride != "" )
    {
        sname = m_dmStreamOverride;
    }
    updateIfChanged( m_indiP_dmStream, "name", sname );

    { //mutex scope
        std::lock_guard<std::mutex> rlock( m_resultsMutex );

        pcf::IndiProperty::PropertyStateType rs =
            ( m_runStatus == "error" ) ? INDI_ALERT : ( m_running ? INDI_BUSY : INDI_IDLE );

        updateIfChanged( m_indiP_runState, "status", m_runStatus, rs );
        updateIfChanged( m_indiP_runState, "phase", m_runPhase, rs );
        updateIfChanged( m_indiP_patternInfo, "info", m_patternInfo );
        updateIfChanged( m_indiP_output, "dir", m_runDir );
        updateIfChanged( m_indiP_delays, "values", m_delaysText );
        updateIfChanged( m_indiP_results, "t50", m_t50Text );
        updateIfChanged( m_indiP_results, "rise", m_riseText );
        updateIfChanged( m_indiP_results, "jitter", m_jitterText );
        updateIfChanged( m_indiP_best,
                         std::vector<std::string>( { "delay_us", "t50", "rise", "jitter" } ),
                         m_bestValues );
    }

    updateIfChanged( m_indiP_progress,
                     std::vector<std::string>( { "delay_index", "delay_us", "sign", "trial", "n_invalid", "late_frac" } ),
                     std::vector<double>( { static_cast<double>( m_progDelayIdx ),
                                            static_cast<double>( m_progDelayUs ),
                                            static_cast<double>( m_progSign ),
                                            static_cast<double>( m_progTrial ),
                                            static_cast<double>( m_progInvalid ),
                                            static_cast<double>( m_progLateFrac ) } ) );

    if( m_running )
    {
        updateSwitchIfChanged( m_indiP_start, "request", pcf::IndiElement::Off, INDI_BUSY );
    }
    else
    {
        updateSwitchIfChanged( m_indiP_start, "request", pcf::IndiElement::Off, INDI_IDLE );
    }

    return 0;
}

inline int dmTemporalResponse::appShutdown()
{
    m_stopRequested = true;

    if( m_measThread.joinable() )
    {
        // Wake the thread if it is waiting for a start request.
        sem_post( &m_startSem );

        try
        {
            m_measThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }

    zeroDM();

    SHMIMMONITORT_APP_SHUTDOWN( shmimMonitorT );

    return 0;
}

inline const std::string &dmTemporalResponse::wfsCamDevName() const
{
    return m_wfsCamDevName;
}

inline int dmTemporalResponse::dmIndex() const
{
    return m_dmIndex;
}

inline int dmTemporalResponse::dmChannel() const
{
    return m_dmChannel;
}

inline const std::string &dmTemporalResponse::pokeModeName() const
{
    return m_pokeModeName;
}

inline const std::vector<int> &dmTemporalResponse::pokeX() const
{
    return m_pokeX;
}

inline const std::vector<int> &dmTemporalResponse::pokeY() const
{
    return m_pokeY;
}

inline const std::string &dmTemporalResponse::patternFile() const
{
    return m_patternFile;
}

inline float dmTemporalResponse::pokeAmp() const
{
    return m_pokeAmp;
}

inline float dmTemporalResponse::maxCommand() const
{
    return m_maxCommand;
}

inline int dmTemporalResponse::nDelays() const
{
    return m_nDelays;
}

inline double dmTemporalResponse::delaySpan() const
{
    return m_delaySpan;
}

inline int dmTemporalResponse::nFrames() const
{
    return m_nFrames;
}

inline int dmTemporalResponse::nTrials() const
{
    return m_nTrials;
}

inline double dmTemporalResponse::settle() const
{
    return m_settle;
}

inline double dmTemporalResponse::trialTimeout() const
{
    return m_trialTimeout;
}

inline int dmTemporalResponse::maxRetries() const
{
    return m_maxRetries;
}

inline int dmTemporalResponse::nRef() const
{
    return m_nRef;
}

inline int dmTemporalResponse::nSettle() const
{
    return m_nSettle;
}

inline float dmTemporalResponse::maskThresh() const
{
    return m_maskThresh;
}

inline int dmTemporalResponse::resampleFactor() const
{
    return m_resampleFactor;
}

inline const std::string &dmTemporalResponse::bestMetric() const
{
    return m_bestMetric;
}

inline double dmTemporalResponse::maxLateFrac() const
{
    return m_maxLateFrac;
}

inline const std::string &dmTemporalResponse::baseDir() const
{
    return m_baseDir;
}

inline int dmTemporalResponse::allocate( const dmTemporalResponseWfsShmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_trialMutex );

        if( m_running )
        {
            m_camChanged = true;
        }

        m_nx     = shmimMonitorT::m_width;
        m_ny     = shmimMonitorT::m_height;
        m_pixget = getPixPointer<float>( shmimMonitorT::m_dataType );
    }

    if( m_pixget == nullptr )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "unsupported camWFS data type" } );
    }

    return 0;
}

inline int dmTemporalResponse::processImage( void *curr_src, const dmTemporalResponseWfsShmimT &dummy )
{
    static_cast<void>( dummy ); // be unused

    return processFrame( curr_src, shmimMonitorT::m_imageStream.md[0].atime, shmimMonitorT::m_imageStream.md[0].cnt0 );
}

inline int dmTemporalResponse::processFrame( void *src, const timespec &atime, uint64_t cnt0 )
{
    m_lastATimeNs = dmTemporalResponseMath::tsToNs( atime );

    if( m_trialState == trialState::idle )
    {
        return 0;
    }

    std::lock_guard<std::mutex> lock( m_trialMutex );

    // Re-check under the lock: the measurement thread may have cancelled the trial.
    if( m_trialState == trialState::idle )
    {
        return 0;
    }

    if( m_trialState == trialState::armed )
    {
        m_trigATime = atime;
        m_tPoke     = dmTemporalResponseMath::tsAddUs( atime, m_curDelay );
        m_trialCnt0 = cnt0;
        m_trialState = trialState::waitPoke;
    }
    else if( m_trialState == trialState::waitPoke )
    {
        if( cnt0 != m_trialCnt0 + 1 )
        {
            m_trialValid = false;
        }
        m_trialCnt0 = cnt0;
    }

    if( m_trialState == trialState::waitPoke )
    {
        // If the deadline falls at or after the next expected frame, record this frame as the baseline and wait for a
        // later frame.
        if( m_framePeriodUs > 0 && dmTemporalResponseMath::tsDiffUs( m_tPoke, atime ) >= m_framePeriodUs )
        {
            copyFrame( src, 0, atime );
            return 0;
        }

        timespec now = m_clock();

        m_trialLate = ( dmTemporalResponseMath::tsDiffUs( now, m_tPoke ) > 0 );

        while( dmTemporalResponseMath::tsDiffUs( now, m_tPoke ) < 0 )
        {
            if( m_stopRequested || m_shutdown )
            {
                m_trialValid = false;
                m_trialState = trialState::idle;
                sem_post( &m_trialSem );
                return 0;
            }

            now = m_clock();
        }

        if( writeDM( ( m_curSign > 0 ) ? m_cmdPos : m_cmdNeg ) < 0 )
        {
            m_trialValid = false;
            m_trialState = trialState::idle;
            sem_post( &m_trialSem );
            return log<software_error, -1>( { __FILE__, __LINE__, "error writing poke to DM" } );
        }

        m_tCmd = m_clock();

        // The baseline copy is done after the poke to keep it off the timing-critical path.  The frame's slot in the
        // circular buffer remains valid until the camera wraps around.
        copyFrame( src, 0, atime );

        m_nCaptured  = 0;
        m_trialState = trialState::capturing;

        return 0;
    }

    // capturing
    if( cnt0 != m_trialCnt0 + 1 )
    {
        m_trialValid = false;
        m_trialState = trialState::idle;
        sem_post( &m_trialSem );
        return 0;
    }
    m_trialCnt0 = cnt0;

    ++m_nCaptured;

    copyFrame( src, m_nCaptured, atime );

    if( m_nCaptured >= m_curNFrames )
    {
        m_trialState = trialState::idle;
        if( sem_post( &m_trialSem ) < 0 )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
        }
    }

    return 0;
}

inline void dmTemporalResponse::copyFrame( void *src, int plane, const timespec &atime )
{
    size_t nPix = static_cast<size_t>( m_nx ) * m_ny;
    float *dst  = m_trialBuf.data() + nPix * plane;

    for( size_t nn = 0; nn < nPix; ++nn )
    {
        dst[nn] = m_pixget( src, nn );
    }

    m_trialTimes[plane] = atime;
}

inline int dmTemporalResponse::writeDM( const mx::improc::eigenImage<float> &cmd )
{
    try
    {
        m_dmStream = cmd;
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception writing DM: " ) + e.what() } );
    }

    return 0;
}

inline void dmTemporalResponse::zeroDM()
{
    if( !m_dmStream.valid() || m_cmdZero.size() == 0 )
    {
        return;
    }

    writeDM( m_cmdZero );
}

inline void dmTemporalResponse::snapshotParams( runParams &params ) const
{
    params.m_dmIndex   = m_dmIndex;
    params.m_dmChannel = m_dmChannel;
    dmTemporalResponseMath::parsePokeMode( params.m_pokeMode, m_pokeModeName );
    params.m_pokeX          = m_pokeX;
    params.m_pokeY          = m_pokeY;
    params.m_patternFile    = m_patternFile;
    params.m_pokeAmp        = m_pokeAmp;
    params.m_maxCommand     = m_maxCommand;
    params.m_nDelays        = m_nDelays;
    params.m_delaySpan      = m_delaySpan;
    params.m_nFrames        = m_nFrames;
    params.m_nTrials        = m_nTrials;
    params.m_settle         = m_settle;
    params.m_trialTimeout   = m_trialTimeout;
    params.m_maxRetries     = m_maxRetries;
    params.m_nRef           = m_nRef;
    params.m_nSettle        = m_nSettle;
    params.m_maskThresh     = m_maskThresh;
    params.m_resampleFactor = m_resampleFactor;
    params.m_bestMetric     = m_bestMetric;
    params.m_maxLateFrac    = m_maxLateFrac;
    params.m_baseDir        = m_baseDir;
    params.m_fps            = m_wfsFps;
}

inline int dmTemporalResponse::prepareRun( const timespec &runStart )
{
    using namespace dmTemporalResponseMath;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_paramMutex );
        snapshotParams( m_run );
    }

    if( m_nx == 0 || m_ny == 0 || m_pixget == nullptr )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "camWFS stream is not connected" } );
    }

    // Clock-domain check: atime must be CLOCK_REALTIME
    if( m_lastATimeNs == 0 || std::fabs( tsDiffUs( m_clock(), nsToTs( m_lastATimeNs ) ) ) > 1e6 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "camWFS atime is not current CLOCK_REALTIME (or no frames): refusing to start" } );
    }

    if( validateTrials( m_run.m_nTrials ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "nTrials must be even and >= 2" } );
    }

    if( m_run.m_nFrames < 1 || m_run.m_nRef < 1 || m_run.m_nSettle < 1 || m_run.m_nSettle > m_run.m_nFrames ||
        m_run.m_resampleFactor < 1 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "nFrames, nRef, nSettle (<= nFrames), and resampleFactor must be >= 1" } );
    }

    std::string bestMetric = m_run.m_bestMetric;
    if( bestMetric != "jitter" && bestMetric != "rise" && bestMetric != "t50" )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "bestMetric must be jitter, rise, or t50" } );
    }

    double span;
    if( resolveSpan( span, m_run.m_delaySpan, m_run.m_fps ) < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "camWFS fps unknown: set delaySpan explicitly or wait for fps" } );
    }

    std::vector<double> delays;
    if( delayGrid( delays, m_run.m_nDelays, span ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid delay grid" } );
    }

    if( validateCommand( m_run.m_pokeAmp, m_run.m_maxCommand ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "poke amplitude must be non-zero and |amp| <= maxCommand" } );
    }

    // Open the DM channel (read-only access to its size; nothing is written until the first trial)
    if( m_dmStreamOverride != "" )
    {
        m_dmStreamName = m_dmStreamOverride;
    }
    else if( dmStreamName( m_dmStreamName, m_run.m_dmIndex, m_run.m_dmChannel ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid DM index or channel" } );
    }

    try
    {
        m_dmStream.open( m_dmStreamName );
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "exception opening DM " + m_dmStreamName + ": " + e.what() } );
    }

    m_dmStream.passive( true );

    uint32_t dmRows = m_dmStream.rows();
    uint32_t dmCols = m_dmStream.cols();

    m_cmdPos.resize( dmRows, dmCols );
    m_cmdNeg.resize( dmRows, dmCols );
    m_cmdZero.resize( dmRows, dmCols );
    m_cmdZero.setZero();

    int x = -1, y = -1;

    if( m_run.m_pokeMode == pokeMode::actuator )
    {
        if( validateActuator( m_run.m_pokeX, m_run.m_pokeY, dmRows, dmCols ) < 0 )
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, "actuator mode requires exactly one in-bounds (x, y)" } );
        }

        x = m_run.m_pokeX[0];
        y = m_run.m_pokeY[0];
        m_patternSha.clear();
    }
    else
    {
        // The pattern is re-read at every run start
        if( loadPattern( m_pattern, m_run.m_patternFile, dmRows, dmCols ) < 0 )
        {
            std::string info = "error: could not load a 2-D " + std::to_string( dmRows ) + "x" +
                               std::to_string( dmCols ) + " pattern from " + m_run.m_patternFile;
            setPatternInfo( info );
            return log<software_error, -1>( { __FILE__, __LINE__, info } );
        }

        if( validatePattern( m_pattern, m_run.m_pokeAmp, m_run.m_maxCommand ) < 0 )
        {
            std::string info = "error: pattern is non-finite, all zero, or amp*pattern exceeds maxCommand";
            setPatternInfo( info );
            return log<software_error, -1>( { __FILE__, __LINE__, info } );
        }

        if( sha256File( m_patternSha, m_run.m_patternFile ) < 0 )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "could not hash pattern file" } );
        }

        setPatternInfo( m_run.m_patternFile + " " + std::to_string( dmRows ) + "x" + std::to_string( dmCols ) +
                        " sha256=" + m_patternSha );
    }

    if( buildPokeCommand( m_cmdPos, m_run.m_pokeMode, x, y, m_pattern, +1, m_run.m_pokeAmp ) < 0 ||
        buildPokeCommand( m_cmdNeg, m_run.m_pokeMode, x, y, m_pattern, -1, m_run.m_pokeAmp ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error building poke commands" } );
    }

    // Create the output directory
    std::string runDir = m_run.m_baseDir + "/" + runDirName( runStart );
    try
    {
        std::filesystem::create_directories( runDir );
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "could not create output directory " + runDir + ": " + e.what() } );
    }

    if( !std::filesystem::is_directory( runDir ) )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "could not create output directory " + runDir } );
    }

    if( m_run.m_pokeMode == pokeMode::pattern )
    {
        try
        {
            std::filesystem::copy_file(
                m_run.m_patternFile, runDir + "/pattern.fits", std::filesystem::copy_options::overwrite_existing );
        }
        catch( const std::exception &e )
        {
            return log<software_error, -1>(
                { __FILE__, __LINE__, std::string( "could not copy pattern to output: " ) + e.what() } );
        }
    }

    m_delays = delays;
    m_span   = span;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_trialMutex );

        m_curNFrames    = m_run.m_nFrames;
        m_framePeriodUs = ( m_run.m_fps > 0 ) ? 1e6 / m_run.m_fps : -1;
        m_trialBuf.resize( m_nx, m_ny, m_run.m_nFrames + 1 );
        m_trialTimes.resize( m_run.m_nFrames + 1 );
    }

    std::string dstr;
    for( size_t k = 0; k < delays.size(); ++k )
    {
        dstr += ( k > 0 ? "," : "" ) + std::to_string( delays[k] );
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_resultsMutex );
        m_runDir     = runDir;
        m_delaysText = dstr;
    }

    return 0;
}

inline int dmTemporalResponse::runTrial( double delay, int sign )
{
    zeroDM();

    // Interruptible settle
    double waited = 0;
    while( waited < m_run.m_settle )
    {
        if( m_stopRequested || m_shutdown || m_camChanged || m_fpsChanged )
        {
            return resultStopped;
        }

        double dt = std::min( 0.01, m_run.m_settle - waited );
        mx::sys::microSleep( static_cast<unsigned>( dt * 1e6 ) );
        waited += dt;
    }

    // Flush any stale posts
    while( sem_trywait( &m_trialSem ) == 0 )
    {
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_trialMutex );

        m_curDelay   = delay;
        m_curSign    = sign;
        m_progDelayUs = delay;
        m_progSign    = sign;
        m_trialValid = true;
        m_trialLate  = false;
        m_nCaptured  = 0;
        m_trialState = trialState::armed;
    }

    timespec start = dmTemporalResponseMath::realtimeNow();
    bool     done  = false;

    while( !done )
    {
        timespec ts = dmTemporalResponseMath::tsAddUs( dmTemporalResponseMath::realtimeNow(), 1e5 );

        if( sem_timedwait( &m_trialSem, &ts ) == 0 )
        {
            done = true;
            break;
        }

        if( m_stopRequested || m_shutdown || m_camChanged || m_fpsChanged ||
            dmTemporalResponseMath::tsDiffUs( dmTemporalResponseMath::realtimeNow(), start ) >
                m_run.m_trialTimeout * 1e6 )
        {
            break;
        }
    }

    int rv = resultValid;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_trialMutex );

        if( !done )
        {
            rv = ( m_stopRequested || m_shutdown || m_camChanged || m_fpsChanged ) ? resultStopped : resultTimeout;
        }
        else if( !m_trialValid )
        {
            rv = ( m_stopRequested || m_shutdown ) ? resultStopped : resultInvalid;
        }

        m_trialState = trialState::idle;
    }

    zeroDM();

    return rv;
}

inline int dmTemporalResponse::runTrialSet( double                            delay,
                                            int                               nPerSign,
                                            bool                              project,
                                            mx::improc::eigenCube<double>    &sumPos,
                                            mx::improc::eigenCube<double>    &sumNeg,
                                            std::vector<std::vector<double>> &curves,
                                            std::vector<std::vector<double>> &times,
                                            std::vector<double>              &delayErrs,
                                            int                              &nLate )
{
    size_t nPix = static_cast<size_t>( m_nx ) * m_ny;
    int    N    = m_run.m_nFrames;

    sumPos.resize( m_nx, m_ny, N );
    sumNeg.resize( m_nx, m_ny, N );
    std::fill( sumPos.data(), sumPos.data() + nPix * N, 0.0 );
    std::fill( sumNeg.data(), sumNeg.data() + nPix * N, 0.0 );

    curves.clear();
    times.clear();
    delayErrs.clear();
    nLate         = 0;
    m_progInvalid = 0;

    for( int sign : { +1, -1 } )
    {
        int nValid = 0;

        while( nValid < nPerSign )
        {
            m_progTrial = nValid;

            int rv = runTrial( delay, sign );

            if( rv == resultStopped )
            {
                return -1;
            }

            if( rv == resultTimeout )
            {
                return log<software_error, -1>( { __FILE__, __LINE__, "trial timed out: is camWFS running?" } );
            }

            if( rv == resultInvalid )
            {
                ++m_progInvalid;

                if( m_progInvalid > m_run.m_maxRetries )
                {
                    return log<software_error, -1>( { __FILE__, __LINE__, "too many invalid trials (frame gaps)" } );
                }

                continue;
            }

            // Valid: accumulate.  The RT thread is idle so the buffer is ours.
            mx::improc::eigenCube<double> &sum = ( sign > 0 ) ? sumPos : sumNeg;

            for( int k = 0; k < N; ++k )
            {
                const float *src = m_trialBuf.data() + nPix * ( k + 1 );
                double      *dst = sum.data() + nPix * k;

                for( size_t nn = 0; nn < nPix; ++nn )
                {
                    dst[nn] += src[nn];
                }
            }

            delayErrs.push_back( dmTemporalResponseMath::achievedDelay( m_trigATime, m_tCmd ) - delay );

            if( m_trialLate )
            {
                ++nLate;
            }

            if( project )
            {
                std::vector<double> r( N ), t( N );

                for( int k = 0; k < N; ++k )
                {
                    r[k] = sign * dmTemporalResponseMath::projectResponse( m_trialBuf.data() + nPix * ( k + 1 ),
                                                                           m_trialBuf.data(),
                                                                           m_refP.data(),
                                                                           m_refMask.data(),
                                                                           nPix,
                                                                           m_refNorm );

                    t[k] = dmTemporalResponseMath::tsDiffUs( m_trialTimes[k + 1], m_tCmd );
                }

                curves.push_back( r );
                times.push_back( t );
            }

            ++nValid;
        }
    }

    return 0;
}

inline int dmTemporalResponse::runReference()
{
    setRunState( "running", "reference" );

    mx::improc::eigenCube<double>    sumPos, sumNeg;
    std::vector<std::vector<double>> curves, times;
    std::vector<double>              delayErrs;
    int                              nLate;

    if( runTrialSet( 0, m_run.m_nRef, false, sumPos, sumNeg, curves, times, delayErrs, nLate ) < 0 )
    {
        return -1;
    }

    mx::improc::eigenCube<float> diff;
    if( dmTemporalResponseMath::differenceCube( diff, sumPos, sumNeg, 2 * m_run.m_nRef ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "error forming reference difference" } );
    }

    // Steady state: average of the last nSettle frames
    m_refP.resize( m_nx, m_ny );
    m_refP.setZero();
    for( int k = m_run.m_nFrames - m_run.m_nSettle; k < m_run.m_nFrames; ++k )
    {
        m_refP += diff.image( k );
    }
    m_refP /= static_cast<float>( m_run.m_nSettle );

    if( dmTemporalResponseMath::buildMask( m_refMask, m_refNorm, m_refP, m_run.m_maskThresh ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "reference pattern is zero: no WFS response to poke" } );
    }

    if( writeReference() < 0 )
    {
        return -1;
    }

    try
    {
        m_refStream.create( m_configName + "_ref", m_nx, m_ny );
        m_refStream = m_refP;
    }
    catch( const std::exception &e )
    {
        log<software_error>( { __FILE__, __LINE__, std::string( "live ref shmim: " ) + e.what() } );
    }

    return 0;
}

inline int dmTemporalResponse::runMeasurement()
{
    using namespace dmTemporalResponseMath;

    m_stopRequested = false;
    m_fpsChanged    = false;
    m_camChanged    = false;
    m_haveBest      = false;
    m_metrics.clear();
    m_respMean.clear();
    m_respStd.clear();
    m_respTime.clear();
    setRunState( "running", "none" );

    timespec runStart = realtimeNow();

    if( prepareRun( runStart ) < 0 )
    {
        setRunState( "error", "none" );
        return -1;
    }

    log<text_log>( "starting DM temporal response run on " + m_dmStreamName + " into " + m_runDir,
                   logPrio::LOG_NOTICE );

    // Always leave the DM channel at zero, however we exit
    struct zeroGuard
    {
        dmTemporalResponse *m_app; ///< The app whose DM is zeroed.

        /// Zero the DM on scope exit.
        ~zeroGuard()
        {
            m_app->zeroDM();
        }
    } guard{ this };

    if( runReference() < 0 )
    {
        setRunState( m_stopRequested ? "stopped" : "error", "none" );
        return -1;
    }

    setRunState( "running", "measuring" );

    int N = m_run.m_nFrames;
    int M = m_run.m_nTrials;

    for( size_t k = 0; k < m_delays.size(); ++k )
    {
        m_progDelayIdx = k;

        mx::improc::eigenCube<double>    sumPos, sumNeg;
        std::vector<std::vector<double>> curves, times;
        std::vector<double>              delayErrs;
        int                              nLate;

        if( runTrialSet( m_delays[k], M / 2, true, sumPos, sumNeg, curves, times, delayErrs, nLate ) < 0 )
        {
            setRunState( m_stopRequested ? "stopped" : "error", "none" );
            return -1;
        }

        mx::improc::eigenCube<float> cube;
        if( differenceCube( cube, sumPos, sumNeg, M ) < 0 )
        {
            setRunState( "error", "none" );
            return log<software_error, -1>( { __FILE__, __LINE__, "error forming difference cube" } );
        }

        // Curve statistics over the M trials
        std::vector<double> rmean( N, 0.0 ), rstd( N, 0.0 ), tmean( N, 0.0 );
        for( size_t j = 0; j < curves.size(); ++j )
        {
            for( int i = 0; i < N; ++i )
            {
                rmean[i] += curves[j][i];
                tmean[i] += times[j][i];
            }
        }
        for( int i = 0; i < N; ++i )
        {
            rmean[i] /= curves.size();
            tmean[i] /= curves.size();
        }
        for( size_t j = 0; j < curves.size(); ++j )
        {
            for( int i = 0; i < N; ++i )
            {
                rstd[i] += ( curves[j][i] - rmean[i] ) * ( curves[j][i] - rmean[i] );
            }
        }
        for( int i = 0; i < N; ++i )
        {
            rstd[i] = sqrt( rstd[i] / curves.size() );
        }

        responseMetrics met;
        if( computeMetrics( met, tmean, rmean, rstd, m_run.m_nSettle, delayErrs, nLate ) < 0 )
        {
            log<text_log>( "delay " + std::to_string( m_delays[k] ) +
                               " us: response did not cross 10/50/90%; t50/rise/jitter unavailable",
                           logPrio::LOG_WARNING );
        }

        m_progLateFrac = met.m_lateFrac;

        if( met.m_lateFrac > m_run.m_maxLateFrac )
        {
            log<text_log>( "delay " + std::to_string( m_delays[k] ) + " us: late-poke fraction " +
                               std::to_string( met.m_lateFrac ) + " exceeds maxLateFrac; minimum achievable delay is " +
                               std::to_string( met.m_delayErrMean + m_delays[k] ) + " us",
                           logPrio::LOG_WARNING );
        }

        m_metrics.push_back( met );
        m_respMean.push_back( rmean );
        m_respStd.push_back( rstd );
        m_respTime.push_back( tmean );

        if( writeCube( k, cube, delayErrs, m_progInvalid ) < 0 )
        {
            setRunState( "error", "none" );
            return -1;
        }

        log<text_log>( "delay " + std::to_string( m_delays[k] ) + " us: t50=" + std::to_string( met.m_t50 ) +
                       " rise=" + std::to_string( met.m_rise ) + " jitter=" + std::to_string( met.m_jitter ) +
                       " delayErr=" + std::to_string( met.m_delayErrMean ) + "+/-" + std::to_string( met.m_delayErrStd ) );

        // Live response curves, N x K (completed delays so far)
        try
        {
            mx::improc::eigenImage<float> resp( N, m_delays.size() );
            resp.setZero();
            for( size_t kk = 0; kk < m_respMean.size(); ++kk )
            {
                for( int i = 0; i < N; ++i )
                {
                    resp( i, kk ) = m_respMean[kk][i];
                }
            }
            if( !m_respStream.valid() || m_respStream.rows() != static_cast<uint32_t>( N ) ||
                m_respStream.cols() != m_delays.size() )
            {
                m_respStream.create( m_configName + "_resp", N, m_delays.size() );
            }
            m_respStream = resp;
        }
        catch( const std::exception &e )
        {
            log<software_error>( { __FILE__, __LINE__, std::string( "live resp shmim: " ) + e.what() } );
        }
    }

    setRunState( "running", "analyzing" );

    // The super-sampling bin is a fraction of the frame period, or of the span if the fps is unknown.
    std::vector<double> grid, val;
    double              T = ( m_framePeriodUs > 0 ) ? m_framePeriodUs : m_span;

    if( resampleAverage( grid, val, m_respTime, m_respMean, T / m_run.m_resampleFactor ) < 0 )
    {
        grid.clear();
        val.clear();
    }

    m_haveBest = ( bestDelay( m_bestIdx, m_metrics, m_run.m_bestMetric ) == 0 );

    std::string t50s, rises, jitters;
    for( size_t k = 0; k < m_metrics.size(); ++k )
    {
        t50s += ( k > 0 ? "," : "" ) + std::to_string( m_metrics[k].m_t50 );
        rises += ( k > 0 ? "," : "" ) + std::to_string( m_metrics[k].m_rise );
        jitters += ( k > 0 ? "," : "" ) + std::to_string( m_metrics[k].m_jitter );
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_resultsMutex );

        m_t50Text    = t50s;
        m_riseText   = rises;
        m_jitterText = jitters;

        if( m_haveBest )
        {
            m_bestValues = { m_delays[m_bestIdx],
                             m_metrics[m_bestIdx].m_t50,
                             m_metrics[m_bestIdx].m_rise,
                             m_metrics[m_bestIdx].m_jitter };
        }
        else
        {
            m_bestValues = { -1, -1, -1, -1 };
        }
    }

    if( m_haveBest )
    {
        log<text_log>( "best delay (" + m_run.m_bestMetric + "): " + std::to_string( m_delays[m_bestIdx] ) + " us",
                       logPrio::LOG_NOTICE );
    }

    if( writeSummary( grid, val ) < 0 )
    {
        setRunState( "error", "none" );
        return -1;
    }

    if( grid.size() > 0 )
    {
        try
        {
            mx::improc::eigenImage<float> ra( grid.size(), 2 );
            for( size_t i = 0; i < grid.size(); ++i )
            {
                ra( i, 0 ) = grid[i];
                ra( i, 1 ) = val[i];
            }
            m_respAvgStream.create( m_configName + "_respavg", grid.size(), 2 );
            m_respAvgStream = ra;
        }
        catch( const std::exception &e )
        {
            log<software_error>( { __FILE__, __LINE__, std::string( "live respavg shmim: " ) + e.what() } );
        }
    }

    setRunState( "done", "none" );

    log<text_log>( "DM temporal response run complete: " + m_runDir, logPrio::LOG_NOTICE );

    return 0;
}

inline int dmTemporalResponse::requestStart()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_paramMutex );

        if( m_running )
        {
            return log<text_log, -1>( "start rejected: a run is already in progress", logPrio::LOG_WARNING );
        }

        m_running = true;
    }

    if( sem_post( &m_startSem ) < 0 )
    {
        m_running = false;
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
    }

    return 0;
}

inline void dmTemporalResponse::requestStop()
{
    m_stopRequested = true;
}

inline void dmTemporalResponse::setRunState( const std::string &status, const std::string &phase )
{
    std::lock_guard<std::mutex> lock( m_resultsMutex );

    m_runStatus = status;
    m_runPhase  = phase;
}

inline void dmTemporalResponse::setPatternInfo( const std::string &info )
{
    std::lock_guard<std::mutex> lock( m_resultsMutex );

    m_patternInfo = info;
}

inline void dmTemporalResponse::appendRunHeader( mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> &fh )
{
    timespec now = dmTemporalResponseMath::realtimeNow();

    fh.append( "DATE-OBS", dmTemporalResponseMath::isoDate( now ), "UTC date of file creation" );
    fh.append( "INSTRUME", std::string( "MagAO-X " ) + m_configName );
    fh.append( "DMSTREAM", m_dmStreamName, "DM channel poked" );
    fh.append( "WFSSHMIM", shmimMonitorT::m_shmimName, "WFS camera stream" );
    fh.append( "WFSFPS", m_run.m_fps, "WFS camera fps at run start" );
    fh.append( "POKEMODE",
               std::string( m_run.m_pokeMode == dmTemporalResponseMath::pokeMode::actuator ? "actuator" : "pattern" ),
               "poke mode" );

    if( m_run.m_pokeMode == dmTemporalResponseMath::pokeMode::actuator )
    {
        fh.append( "POKEX", m_run.m_pokeX[0], "actuator x" );
        fh.append( "POKEY", m_run.m_pokeY[0], "actuator y" );
    }
    else
    {
        fh.append( "PATFILE", m_run.m_patternFile, "DM pattern file" );
        fh.append( "PATSHA", m_patternSha, "SHA-256 of pattern file" );
    }

    fh.append( "POKEAMP", m_run.m_pokeAmp, "poke amplitude [DM units]" );
    fh.append( "NDELAYS", m_run.m_nDelays, "K, number of delays" );
    fh.append( "DLYSPAN", m_span, "delay grid span [us]" );
    fh.append( "NFRAMES", m_run.m_nFrames, "N, frames per trial" );
    fh.append( "NTRIALS", m_run.m_nTrials, "M, trials per delay (M/2 +, M/2 -)" );
    fh.append( "NREF", m_run.m_nRef, "reference-pass trial pairs" );
}

inline int dmTemporalResponse::writeCube( size_t                              idx,
                                          const mx::improc::eigenCube<float> &cube,
                                          const std::vector<double>          &delayErrs,
                                          int                                 nInvalid )
{
    mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> fh;

    appendRunHeader( fh );

    const dmTemporalResponseMath::responseMetrics &met = m_metrics[idx];

    double dmin = 0, dmax = 0;
    if( delayErrs.size() > 0 )
    {
        dmin = *std::min_element( delayErrs.begin(), delayErrs.end() ) + m_delays[idx];
        dmax = *std::max_element( delayErrs.begin(), delayErrs.end() ) + m_delays[idx];
    }

    fh.append( "DELAYUS", m_delays[idx], "requested delay [us]" );
    fh.append( "DLYIDX", static_cast<int>( idx ), "delay index" );
    fh.append( "DLYMEAN", met.m_delayErrMean + m_delays[idx], "mean achieved delay [us]" );
    fh.append( "DLYSTD", met.m_delayErrStd, "std of achieved delay [us]" );
    fh.append( "DLYMIN", dmin, "min achieved delay [us]" );
    fh.append( "DLYMAX", dmax, "max achieved delay [us]" );
    fh.append( "NINVALID", nInvalid, "invalid (retried) trials" );
    fh.append( "LATEFRAC", met.m_lateFrac, "fraction of late pokes" );
    fh.append( "LATEFLAG", static_cast<int>( met.m_lateFrac > m_run.m_maxLateFrac ), "1 if LATEFRAC > maxLateFrac" );
    fh.append( "T50", met.m_t50, "t50 [us] from command" );
    fh.append( "RISE", met.m_rise, "10-90% rise [us]" );
    fh.append( "JITTER", met.m_jitter, "std of r at t50 frame" );

    std::string fname = m_runDir + "/" + dmTemporalResponseMath::cubeFileName( m_delays[idx] );

    try
    {
        mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
        if( !dmTemporalResponseMath::writeOk( ff.write( fname, cube, fh ) ) )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "error writing " + fname } );
        }
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "exception writing " + fname + ": " + e.what() } );
    }

    return 0;
}

inline int dmTemporalResponse::writeReference()
{
    mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> fh;

    appendRunHeader( fh );
    fh.append( "MASKTHR", m_run.m_maskThresh, "mask threshold, fraction of max|P|" );
    fh.append( "NSETTLE", m_run.m_nSettle, "trailing frames averaged for P" );
    fh.append( "PLANE0", std::string( "P" ), "reference pattern" );
    fh.append( "PLANE1", std::string( "mask" ), "pixel mask" );

    mx::improc::eigenCube<float> ref( m_nx, m_ny, 2 );
    ref.image( 0 ) = m_refP;
    ref.image( 1 ) = m_refMask;

    std::string fname = m_runDir + "/reference.fits";

    try
    {
        mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;
        if( !dmTemporalResponseMath::writeOk( ff.write( fname, ref, fh ) ) )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "error writing " + fname } );
        }
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "exception writing " + fname + ": " + e.what() } );
    }

    return 0;
}

inline int dmTemporalResponse::writeSummary( const std::vector<double> &grid, const std::vector<double> &val )
{
    size_t K = m_metrics.size();
    int    N = m_run.m_nFrames;

    mx::fits::fitsFile<float, XWC_DEFAULT_VERBOSITY> ff;

    // Curves: [N, K, 3] planes are mean response, std, and time [us]
    mx::improc::eigenCube<float> curves( N, K, 3 );
    for( size_t k = 0; k < K; ++k )
    {
        for( int i = 0; i < N; ++i )
        {
            curves.image( 0 )( i, k ) = m_respMean[k][i];
            curves.image( 1 )( i, k ) = m_respStd[k][i];
            curves.image( 2 )( i, k ) = m_respTime[k][i];
        }
    }

    // Metrics: [K, 9]
    static const std::vector<std::string> cols = {
        "delay", "t50", "rise", "overshoot", "settleErr", "jitter", "delayErrMean", "delayErrStd", "lateFrac" };

    mx::improc::eigenImage<float> metrics( K, cols.size() );
    for( size_t k = 0; k < K; ++k )
    {
        const dmTemporalResponseMath::responseMetrics &m = m_metrics[k];
        metrics( k, 0 ) = m_delays[k];
        metrics( k, 1 ) = m.m_t50;
        metrics( k, 2 ) = m.m_rise;
        metrics( k, 3 ) = m.m_overshoot;
        metrics( k, 4 ) = m.m_settleErr;
        metrics( k, 5 ) = m.m_jitter;
        metrics( k, 6 ) = m.m_delayErrMean;
        metrics( k, 7 ) = m.m_delayErrStd;
        metrics( k, 8 ) = m.m_lateFrac;
    }

    // Super-sampled response: [nGrid, 2] columns are time [us] and response
    mx::improc::eigenImage<float> superres( std::max<size_t>( grid.size(), 1 ), 2 );
    superres.setConstant( std::numeric_limits<float>::quiet_NaN() );
    for( size_t i = 0; i < grid.size(); ++i )
    {
        superres( i, 0 ) = grid[i];
        superres( i, 1 ) = val[i];
    }

    try
    {
        mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> fhc;
        appendRunHeader( fhc );
        fhc.append( "PLANE0", std::string( "rmean" ), "mean response vs frame (rows) and delay (cols)" );
        fhc.append( "PLANE1", std::string( "rstd" ), "trial std of the response" );
        fhc.append( "PLANE2", std::string( "time" ), "mean time from DM command [us]" );
        if( !dmTemporalResponseMath::writeOk( ff.write( m_runDir + "/summary_curves.fits", curves, fhc ) ) )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "error writing summary_curves.fits" } );
        }

        mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> fhm;
        appendRunHeader( fhm );
        for( size_t c = 0; c < cols.size(); ++c )
        {
            fhm.append( "MCOL" + std::to_string( c ), cols[c], "metrics column " + std::to_string( c ) );
        }
        fhm.append( "BESTMET", m_run.m_bestMetric, "best-delay criterion" );
        fhm.append( "BESTIDX", m_haveBest ? static_cast<int>( m_bestIdx ) : -1, "best delay index, -1 if none" );
        fhm.append( "BESTDLY", m_haveBest ? m_delays[m_bestIdx] : -1.0, "best delay [us], -1 if none" );
        if( !dmTemporalResponseMath::writeOk( ff.write( m_runDir + "/summary_metrics.fits", metrics, fhm ) ) )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "error writing summary_metrics.fits" } );
        }

        mx::fits::fitsHeader<XWC_DEFAULT_VERBOSITY> fhs;
        appendRunHeader( fhs );
        fhs.append( "COL0", std::string( "time" ), "bin center time from DM command [us]" );
        fhs.append( "COL1", std::string( "response" ), "binned mean response" );
        fhs.append( "RESAMP", m_run.m_resampleFactor, "super-sampling factor" );
        if( !dmTemporalResponseMath::writeOk( ff.write( m_runDir + "/summary_superres.fits", superres, fhs ) ) )
        {
            return log<software_error, -1>( { __FILE__, __LINE__, "error writing summary_superres.fits" } );
        }
    }
    catch( const std::exception &e )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, std::string( "exception writing summary: " ) + e.what() } );
    }

    return 0;
}

inline void dmTemporalResponse::measThreadStart( dmTemporalResponse *s )
{
    s->measThreadExec();
}

inline void dmTemporalResponse::measThreadExec()
{
    m_measThreadID = syscall( SYS_gettid );

    // Wait for the thread starter to finish initializing this thread.
    while( m_measThreadInit == true && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    while( m_shutdown == 0 )
    {
        timespec ts = dmTemporalResponseMath::tsAddUs( dmTemporalResponseMath::realtimeNow(), 1e6 );

        if( sem_timedwait( &m_startSem, &ts ) != 0 )
        {
            continue;
        }

        if( m_shutdown )
        {
            break;
        }

        runMeasurement();

        m_running = false;
    }
}

template <typename T>
int dmTemporalResponse::tunableCallback( pcf::IndiProperty &local, T &member, const pcf::IndiProperty &ipRecv )
{
    std::lock_guard<std::mutex> lock( m_paramMutex );

    if( m_running )
    {
        return log<text_log, -1>( "change to " + local.getName() + " rejected while a run is in progress",
                                  logPrio::LOG_WARNING );
    }

    T target;

    if( indiTargetUpdate( local, target, ipRecv, false ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    member = target;

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_dmIndex )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dmIndex, ipRecv );

    return tunableCallback( m_indiP_dmIndex, m_dmIndex, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_dmChannel )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_dmChannel, ipRecv );

    return tunableCallback( m_indiP_dmChannel, m_dmChannel, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_pokeMode )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pokeMode, ipRecv );

    std::lock_guard<std::mutex> lock( m_paramMutex );

    if( m_running )
    {
        return log<text_log, -1>( "change to poke_mode rejected while a run is in progress", logPrio::LOG_WARNING );
    }

    for( const std::string &mode : { std::string( "actuator" ), std::string( "pattern" ) } )
    {
        if( ipRecv.find( mode ) && ipRecv[mode].getSwitchState() == pcf::IndiElement::On )
        {
            m_pokeModeName = mode;

            m_indiP_pokeMode["actuator"].setSwitchState( mode == "actuator" ? pcf::IndiElement::On
                                                                            : pcf::IndiElement::Off );
            m_indiP_pokeMode["pattern"].setSwitchState( mode == "pattern" ? pcf::IndiElement::On
                                                                          : pcf::IndiElement::Off );
            return 0;
        }
    }

    return log<text_log, -1>( "poke_mode: no element switched on", logPrio::LOG_WARNING );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_pokeX )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pokeX, ipRecv );

    int x = -1;
    if( tunableCallback( m_indiP_pokeX, x, ipRecv ) < 0 )
    {
        return -1;
    }

    m_pokeX = { x };

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_pokeY )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pokeY, ipRecv );

    int y = -1;
    if( tunableCallback( m_indiP_pokeY, y, ipRecv ) < 0 )
    {
        return -1;
    }

    m_pokeY = { y };

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_patternFile )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_patternFile, ipRecv );

    return tunableCallback( m_indiP_patternFile, m_patternFile, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_pokeAmp )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_pokeAmp, ipRecv );

    return tunableCallback( m_indiP_pokeAmp, m_pokeAmp, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_nDelays )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nDelays, ipRecv );

    return tunableCallback( m_indiP_nDelays, m_nDelays, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_delaySpan )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_delaySpan, ipRecv );

    return tunableCallback( m_indiP_delaySpan, m_delaySpan, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_nFrames )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nFrames, ipRecv );

    return tunableCallback( m_indiP_nFrames, m_nFrames, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_nTrials )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nTrials, ipRecv );

    return tunableCallback( m_indiP_nTrials, m_nTrials, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_settle )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_settle, ipRecv );

    return tunableCallback( m_indiP_settle, m_settle, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_nSettle )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_nSettle, ipRecv );

    return tunableCallback( m_indiP_nSettle, m_nSettle, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_maskThresh )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_maskThresh, ipRecv );

    return tunableCallback( m_indiP_maskThresh, m_maskThresh, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_bestMetric )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_bestMetric, ipRecv );

    return tunableCallback( m_indiP_bestMetric, m_bestMetric, ipRecv );
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_start )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_start, ipRecv );

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        return requestStart();
    }

    return 0;
}

INDI_NEWCALLBACK_DEFN( dmTemporalResponse, m_indiP_stop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_stop, ipRecv );

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        requestStop();
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( dmTemporalResponse, m_indiP_wfsFps )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_wfsFps, ipRecv );

    if( !ipRecv.find( "current" ) )
    {
        return 0;
    }

    double fps = ipRecv["current"].get<double>();

    // An fps change of more than 0.1% during a run invalidates the delay grid.
    if( m_running && m_run.m_fps > 0 && std::fabs( fps - m_run.m_fps ) > 1e-3 * m_run.m_fps )
    {
        m_fpsChanged = true;
        log<text_log>( "camWFS fps changed during run: aborting", logPrio::LOG_WARNING );
    }

    m_wfsFps = fps;

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // dmTemporalResponse_hpp
