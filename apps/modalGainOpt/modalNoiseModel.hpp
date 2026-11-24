/** \file modalNoiseModel.hpp
 * \brief Detector-noise propagation and effective-gain limits for modal gain optimization.
 */

#ifndef modalGainOpt_modalNoiseModel_hpp
#define modalGainOpt_modalNoiseModel_hpp

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <memory>
#include <regex>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <ImageStreamIO/ImageStreamIO.h>
#include <mx/ao/analysis/clAOLinearPredictor.hpp>
#include <mx/ao/analysis/clGainOpt.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/math/floatUtils.hpp>
#include <mx/math/ft/fftT.hpp>
#include <mx/sigproc/signalWindows.hpp>

namespace MagAOX
{
namespace app
{
namespace modalNoise
{

/// Requested or applied source of sampled WFS noise.
enum class Source
{
    empirical,
    camera,
    detector
};

/// Return the INDI name of a noise source.
inline std::string sourceName( Source source /**< [in] Noise source. */ )
{
    switch( source )
    {
    case Source::camera:
        return "camera";
    case Source::detector:
        return "detector";
    default:
        return "empirical";
    }
}

/// Parse a supported INDI noise source.
inline bool parseSource( Source &source /**< [out] Parsed source. */, const std::string &name /**< [in] INDI name. */ )
{
    for( Source value : { Source::camera, Source::detector, Source::empirical } )
    {
        if( name == sourceName( value ) )
        {
            source = value;
            return true;
        }
    }

    return false;
}

/// One inclusive range of zero-based modes, with an optional unbounded end.
struct GainCeiling
{
    size_t m_first; ///< First mode to constrain.
    size_t m_last;  ///< Last mode to constrain or the maximum size_t for an open end.
    float  m_value; ///< Upper bound in effective gain.
};

/// Parse comma-separated first-last:gain entries. An empty last mode extends to the active basis end.
inline bool parseCeilings( std::vector<GainCeiling> &ceilings /**< [out] Validated limits. */,
                           const std::string        &spec /**< [in] Range list or "none". */ )
{
    std::vector<GainCeiling> parsed;

    if( !spec.empty() && spec != "none" )
    {
        static const std::regex entry( R"(\s*([0-9]+)\s*-\s*([0-9]*)\s*:\s*([0-9]*\.?[0-9]+)\s*)" );
        size_t                  start = 0;

        try
        {
            do
            {
                const size_t      end   = spec.find( ',', start );
                const std::string token = spec.substr( start, end - start );
                std::smatch       fields;

                if( !std::regex_match( token, fields, entry ) )
                {
                    return false;
                }

                const auto first = std::stoull( fields[1] );
                const auto last =
                    fields[2].length() == 0 ? std::numeric_limits<size_t>::max() : std::stoull( fields[2] );
                const float value = std::stof( fields[3] );

                if( first > last || last > std::numeric_limits<size_t>::max() || !mx::math::isFinite( value ) ||
                    value <= 0 )
                {
                    return false;
                }

                parsed.push_back( { static_cast<size_t>( first ), static_cast<size_t>( last ), value } );
                start = end + 1;
            } while( start != 0 );
        }
        catch( const std::exception & )
        {
            return false;
        }
    }

    ceilings = std::move( parsed );
    return true;
}

/// Find the tightest configured ceiling for this mode.
inline float ceilingForMode( const std::vector<GainCeiling> &ceilings /**< [in] Configured ranges. */,
                             size_t                          mode /**< [in] Zero-based mode index. */ )
{
    float limit = std::numeric_limits<float>::infinity();

    for( const auto &range : ceilings )
    {
        if( mode >= range.m_first && mode <= range.m_last )
        {
            limit = std::min( limit, range.m_value );
        }
    }

    return limit;
}

/// Minimize the physical residual within both the controller's stability margin and the mode ceiling.
inline mx::error_t
optimizeWithCeiling( float                              &gain /**< [out] Best effective gain. */,
                     float                              &variance /**< [out] Modeled residual variance. */,
                     mx::AO::analysis::clGainOpt<float> &optimizer /**< [in,out] Candidate controller. */,
                     const std::vector<float>           &disturbance /**< [in] DM disturbance PSD. */,
                     const std::vector<float>           &noise /**< [in] Input-referred detector PSD. */,
                     float                               stability /**< [in] Stability gain before margin. */,
                     float                               ceiling /**< [in] Configured effective gain maximum. */ )
{
    const float margin = optimizer.m_minFindMaxFact;

    if( ceiling >= stability * margin )
    {
        return optimizer.optGainOpenLoop( gain, variance, disturbance, noise, stability, false );
    }

    if( ceiling > optimizer.m_minFindMin )
    {
        const auto error = optimizer.optGainOpenLoop( gain, variance, disturbance, noise, ceiling / margin, false );

        if( !!error )
        {
            return error;
        }
    }
    else
    {
        gain     = 0;
        variance = optimizer.clVariance( disturbance, noise, gain );
    }

    const float ceilingVariance = optimizer.clVariance( disturbance, noise, ceiling );

    if( gain >= ceiling || ceilingVariance <= variance )
    {
        gain     = ceiling;
        variance = ceilingVariance;
    }

    return mx::error_t::noerror;
}

/// Tune predictor regularization against the same bounded gain objective used after coefficient updates.
inline mx::error_t regularizeWithCeiling(
    float &stability /**< [out] Stability gain of the selected predictor. */,
    float &gain /**< [out] Best effective predictor gain. */,
    float &variance /**< [out] Modeled residual variance. */,
    float &scale /**< [out] Selected regularization scale in dB. */,
    mx::AO::analysis::clAOLinearPredictor<float>
                                       &predictor /**< [in,out] Linear predictor whose coefficients are tuned. */,
    mx::AO::analysis::clGainOpt<float> &optimizer /**< [in,out] Candidate predictor controller. */,
    std::vector<float>                 &continuum /**< [in] Physical disturbance with any LP continuum smoothing. */,
    const std::vector<float>           &disturbance /**< [in] Disturbance used for the final LP gain objective. */,
    std::vector<float>                 &noise /**< [in] Input-referred detector PSD. */,
    int                                 coefficients /**< [in] Number of predictor coefficients. */,
    float                               ceiling /**< [in] Configured effective gain maximum. */ )
{
    const float originalLimit = predictor.m_gmax_lp;
    const float minimum       = std::nextafter( optimizer.m_minFindMin, 1.0F );
    predictor.m_gmax_lp       = std::min( originalLimit, std::max( ceiling, minimum ) / optimizer.m_minFindMaxFact );
    auto error =
        predictor.regularizeCoefficients( stability, gain, variance, scale, optimizer, continuum, noise, coefficients );
    predictor.m_gmax_lp = originalLimit;

    if( !error && mx::math::isFinite( ceiling ) )
    {
        error = optimizeWithCeiling( gain, variance, optimizer, disturbance, noise, stability, ceiling );
    }

    return error;
}

/// Return the measured open-loop-equivalent log median in the upper 20% of the Nyquist band.
inline float measuredFloor( const float              *psd /**< [in] Raw WFS one-sided PSD. */,
                            const std::vector<float> &freq /**< [in] Frequency grid in Hz. */,
                            const std::vector<float> &etf /**< [in] Current ETF power, or unity for open loop. */,
                            float                     fps /**< [in] Sampling rate in Hz. */,
                            bool halfWeightNyquist = false /**< [in] Whether the producer half weights Nyquist. */ )
{
    if( !psd || etf.size() != freq.size() || !mx::math::isFinite( fps ) || fps <= 0 )
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<float> logs;
    const float        nyquist   = fps / 2;
    const float        tolerance = nyquist * 4 * std::numeric_limits<float>::epsilon();

    for( size_t f = 0; f < freq.size(); ++f )
    {
        if( freq[f] < 0.8F * nyquist - tolerance || freq[f] > nyquist + tolerance ||
            ( halfWeightNyquist && std::abs( freq[f] - nyquist ) <= tolerance ) )
        {
            continue;
        }

        if( mx::math::isFinite( psd[f] ) && psd[f] > 0 && mx::math::isFinite( etf[f] ) && etf[f] > 1e-12F )
        {
            logs.push_back( std::log( psd[f] ) - std::log( etf[f] ) );
        }
    }

    if( logs.empty() )
    {
        return std::numeric_limits<float>::quiet_NaN();
    }

    auto mid = logs.begin() + logs.size() / 2;
    std::nth_element( logs.begin(), mid, logs.end() );
    const float median = logs.size() % 2 ? *mid : ( *std::max_element( logs.begin(), mid ) + *mid ) / 2;
    return std::exp( median );
}

/// Camera input-electron calibration at one measured EM setting.
struct Detector
{
    double m_emg;  ///< Electron multiplication setting.
    double m_gain; ///< Mean input electrons per ADU.
    double m_ron;  ///< Input-referred read RMS per physical image pixel, in electrons.
};

/// Find a measured detector calibration. Unsupported EM settings fall back to camera data.
inline const Detector *detectorAt( double emg /**< [in] Actual camera EM setting. */ )
{
    static constexpr std::array<Detector, 3> table{
        { { 10, 1.1535, 14.51 }, { 100, 0.144, 2.0375 }, { 595, 0.0282, 0.40675 } } };

    for( const auto &entry : table )
    {
        if( emg == entry.m_emg )
        {
            return &entry;
        }
    }

    return nullptr;
}

/// Paths to the matched WFS mask, reference, mode cube, and selfRM cube, in that order.
using CalibrationFiles = std::array<std::filesystem::path, 4>;

/// Matched reconstructor rows, diagonal noise contractions and static selfRM responses.
class Calibration
{
  public:
    /// Reconstructor columns are modes and rows are masked pixels in native image order.
    using Rows = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic>;

    /// Number of modes in the matched basis.
    size_t modes() const;

    /// Validate and load the active WFS mode cube, mask, reference and plane-5 selfRM.
    bool load( const CalibrationFiles &files /**< [in] Matched FITS paths, resolved in the active archive. */,
               double                  floorPerPixel /**< [in] Runtime WFS floor in ADU per full-frame pixel. */,
               double                  responseBeta /**< [in] Flux normalization fraction at selfRM acquisition. */,
               bool                    modeNorm /**< [in] Runtime MODENORM setting. */ );

    /// Compute masked calibration contractions from already normalized reconstructor rows.
    bool set( const Rows           &rows /**< [in] Runtime reconstructor weights. */,
              const Eigen::ArrayXd &fractions /**< [in] Masked reference photon fractions. */,
              const Eigen::ArrayXd &response /**< [in] Static selfRM response by mode. */,
              double                floor /**< [in] Full normalization floor in ADU. */,
              double                responseBeta /**< [in] Flux fraction at selfRM acquisition. */ );

    /// Return the raw WFS photon and read terms or an empty vector when inputs are unavailable.
    std::vector<float> detectorPsd( double          flux /**< [in] Masked dark-subtracted ADU per frame. */,
                                    double          fps /**< [in] Measured sampling rate in Hz. */,
                                    const Detector *detector /**< [in] Measured detector calibration. */ ) const;

    /// Project high-frequency camera pixel variances into one-sided raw WFS PSDs.
    std::vector<float> cameraPsd(
        double                flux /**< [in] Masked dark-subtracted ADU per frame. */,
        double                fps /**< [in] Measured sampling rate in Hz. */,
        const Eigen::ArrayXd &variance /**< [in] Masked white-equivalent pixel variances in ADU squared. */ ) const;

    /// Return the squared live WFS response per unit of DM modal command.
    double responsePower( size_t mode /**< [in] Zero-based mode index. */,
                          double flux /**< [in] Masked dark-subtracted ADU per frame. */,
                          double opticalGain /**< [in] Optical gain relative to the selfRM calibration. */ ) const;

    /// Return whether an image has the active WFS frame dimensions.
    bool matchesFrame( size_t width /**< [in] Image axis 0. */, size_t height /**< [in] Image axis 1. */ ) const;

    /// Return whether the live mask matches the calibration pixel selection.
    bool matchesMask( const float *mask /**< [in] Live mask in native image order. */ ) const;

    /// Return the selected pixel offsets in native image order.
    const std::vector<size_t> &pixels() const;

  private:
    Rows                m_rows;              ///< Runtime WFS weights restricted to the binary mask.
    Eigen::ArrayXd      m_mean;              ///< Reconstructor response to the reference photon fractions.
    Eigen::ArrayXd      m_photon;            ///< Photon contraction at beta zero.
    Eigen::ArrayXd      m_read;              ///< Read contraction at beta zero.
    Eigen::ArrayXd      m_rowSum;            ///< Sum of each masked reconstructor row.
    Eigen::ArrayXd      m_response;          ///< Diagonal of the calibrated modal self-response.
    double              m_floor{ 0 };        ///< Normalization floor in full-frame ADU.
    double              m_responseBeta{ 1 }; ///< Normalized flux fraction at selfRM measurement.
    size_t              m_width{ 0 };        ///< Active calibration image axis 0.
    size_t              m_height{ 0 };       ///< Active calibration image axis 1.
    std::vector<size_t> m_pixels;            ///< Native pixel offsets within the active binary mask.
};

inline size_t Calibration::modes() const
{
    return m_response.size();
}

inline bool Calibration::set( const Rows           &rows,
                              const Eigen::ArrayXd &fractions,
                              const Eigen::ArrayXd &response,
                              double                floor,
                              double                responseBeta )
{
    if( rows.rows() == 0 || rows.cols() == 0 || rows.rows() != fractions.size() || rows.cols() != response.size() ||
        !rows.unaryExpr(
                 []( double value )
                 {
                     return mx::math::isFinite( value );
                 } )
             .all() ||
        !fractions
             .unaryExpr(
                 []( double value )
                 {
                     return mx::math::isFinite( value );
                 } )
             .all() ||
        ( fractions < 0 ).any() || std::abs( fractions.sum() - 1 ) > 1e-6 ||
        !response
             .unaryExpr(
                 []( double value )
                 {
                     return mx::math::isFinite( value );
                 } )
             .all() ||
        ( response <= 0 ).any() || !mx::math::isFinite( floor ) || floor < 0 ||
        !( responseBeta > 0 && responseBeta <= 1 ) )
    {
        return false;
    }

    m_rows         = rows;
    m_mean         = ( rows.matrix().transpose() * fractions.matrix() ).array();
    m_photon       = ( rows.square().matrix().transpose() * fractions.matrix() ).array();
    m_read         = rows.square().colwise().sum().transpose();
    m_rowSum       = rows.colwise().sum().transpose();
    m_response     = response;
    m_floor        = floor;
    m_responseBeta = responseBeta;
    return true;
}

inline bool Calibration::load( const CalibrationFiles &files, double floorPerPixel, double responseBeta, bool modeNorm )
{
    using File = mx::fits::fitsFile<float>;
    File maskFile;
    File refFile;
    File modesFile;
    File rmFile;
    const auto &[maskPath, referencePath, modesPath, selfRm] = files;

    if( !!maskFile.open( maskPath.string() ) || !!refFile.open( referencePath.string() ) ||
        !!modesFile.open( modesPath.string() ) || !!rmFile.open( selfRm.string() ) )
    {
        return false;
    }

    const long     width  = maskFile.naxes( 0 );
    const long     height = maskFile.naxes( 1 );
    const long     count  = modesFile.naxes( 2 );
    constexpr long plane  = 5;

    if( width <= 0 || height <= 0 || count <= 0 || maskFile.naxis() != 2 || refFile.naxis() != 2 ||
        modesFile.naxis() != 3 || rmFile.naxis() != 3 || refFile.naxes( 0 ) != width || refFile.naxes( 1 ) != height ||
        modesFile.naxes( 0 ) != width || modesFile.naxes( 1 ) != height || rmFile.naxes( 0 ) != count ||
        rmFile.naxes( 1 ) != count || rmFile.naxes( 2 ) <= plane )
    {
        return false;
    }

    const size_t       nPixels = width * height;
    std::vector<float> mask( nPixels ), reference( nPixels ), cube( nPixels * count ), responsePlane( count * count );
    rmFile.setCubeReadSize( plane, 1 );

    if( !!maskFile.read( mask.data() ) || !!refFile.read( reference.data() ) || !!modesFile.read( cube.data() ) ||
        !!rmFile.read( responsePlane.data() ) )
    {
        return false;
    }

    std::vector<size_t> pixels;

    for( size_t p = 0; p < nPixels; ++p )
    {
        if( mask[p] == 1 )
        {
            pixels.push_back( p );
        }
        else if( mask[p] != 0 )
        {
            return false;
        }
    }

    Eigen::ArrayXd fractions( pixels.size() ), response( count );
    Rows           rows( pixels.size(), count );

    for( size_t p = 0; p < pixels.size(); ++p )
    {
        fractions[p] = reference[pixels[p]];
    }

    if( !( fractions.sum() > 0 ) )
    {
        return false;
    }

    fractions /= fractions.sum();

    for( long mode = 0; mode < count; ++mode )
    {
        const Eigen::Map<const Eigen::ArrayXf> image( cube.data() + mode * nPixels, nPixels );
        const double                           norm = modeNorm ? image.cast<double>().square().sum() : 1;

        if( !image
                 .unaryExpr(
                     []( float value )
                     {
                         return mx::math::isFinite( value );
                     } )
                 .all() ||
            !( norm > 0 ) )
        {
            return false;
        }

        for( size_t p = 0; p < pixels.size(); ++p )
        {
            rows( p, mode ) = image[pixels[p]] / norm;
        }

        response[mode] = responsePlane[mode * count + mode];
    }

    if( !set( rows, fractions, response, floorPerPixel * nPixels, responseBeta ) )
    {
        return false;
    }

    m_width  = width;
    m_height = height;
    m_pixels = std::move( pixels );
    return true;
}

inline std::vector<float> Calibration::detectorPsd( double flux, double fps, const Detector *detector ) const
{
    if( !detector || !mx::math::isFinite( flux ) || flux <= 0 || !mx::math::isFinite( fps ) || fps <= 0 ||
        modes() == 0 )
    {
        return {};
    }

    const double         beta   = flux / ( flux + m_floor );
    const double         excess = 2 - 1 / detector->m_emg;
    const double         scale  = 2 / ( fps * std::pow( ( flux + m_floor ) * detector->m_gain, 2 ) );
    const Eigen::ArrayXd shift  = beta * m_mean;
    const Eigen::ArrayXd photon = ( m_photon - 2 * shift * m_mean + shift.square() ).max( 0 );
    const Eigen::ArrayXd read   = ( m_read - 2 * shift * m_rowSum + m_rows.rows() * shift.square() ).max( 0 );
    const Eigen::ArrayXf psd =
        ( scale * ( excess * flux * detector->m_gain * photon + detector->m_ron * detector->m_ron * read ) )
            .cast<float>();
    return { psd.data(), psd.data() + psd.size() };
}

inline std::vector<float> Calibration::cameraPsd( double flux, double fps, const Eigen::ArrayXd &variance ) const
{
    if( !mx::math::isFinite( flux ) || flux <= 0 || !mx::math::isFinite( fps ) || fps <= 0 || modes() == 0 ||
        variance.size() != m_rows.rows() ||
        !variance
             .unaryExpr(
                 []( double value )
                 {
                     return mx::math::isFinite( value );
                 } )
             .all() ||
        ( variance < 0 ).any() )
    {
        return {};
    }

    const double       beta  = flux / ( flux + m_floor );
    const double       scale = 2 / ( fps * std::pow( flux + m_floor, 2 ) );
    std::vector<float> psd( modes() );

    for( size_t mode = 0; mode < psd.size(); ++mode )
    {
        psd[mode] = scale * ( ( m_rows.col( mode ) - beta * m_mean[mode] ).square() * variance ).sum();
    }

    return psd;
}

inline double Calibration::responsePower( size_t mode, double flux, double opticalGain ) const
{
    return std::pow( m_response[mode] * opticalGain * ( flux / ( flux + m_floor ) ) / m_responseBeta, 2 );
}

inline bool Calibration::matchesFrame( size_t width, size_t height ) const
{
    return modes() > 0 && width == m_width && height == m_height;
}

inline bool Calibration::matchesMask( const float *mask ) const
{
    size_t next = 0;

    for( size_t p = 0; p < m_width * m_height; ++p )
    {
        const bool included = next < m_pixels.size() && m_pixels[next] == p;

        if( mask[p] != static_cast<float>( included ) )
        {
            return false;
        }

        next += included;
    }

    return true;
}

inline const std::vector<size_t> &Calibration::pixels() const
{
    return m_pixels;
}

/// Convert an acquisition or publication timestamp to fractional Unix seconds.
inline double seconds( const timespec &time /**< [in] Timestamp to convert. */ )
{
    return time.tv_sec + 1e-9 * time.tv_nsec;
}

/// Estimate white-equivalent pixel variance from the upper 20% of the sampled band.
class CameraNoise
{
  public:
    /// Copy contiguous camera blocks from a buffered raw WFS stream and estimate their pixel variance.
    Eigen::ArrayXd estimate( const std::string &name /**< [in] Raw camera circular buffer name. */,
                             const Calibration &calibration /**< [in] Matched WFS pixel selection. */,
                             double             start /**< [in] Start of the modal PSD interval in Unix seconds. */,
                             double             end /**< [in] End of the modal PSD interval in Unix seconds. */,
                             double             fps /**< [in] Measured camera sampling rate. */ );

    /// Estimate pixel variance from an already attached camera buffer.
    Eigen::ArrayXd estimate( const IMAGE       &image /**< [in] Timestamped raw camera history. */,
                             const Calibration &calibration /**< [in] Matched WFS pixel selection. */,
                             double             start /**< [in] Start of the modal PSD interval in Unix seconds. */,
                             double             end /**< [in] End of the modal PSD interval in Unix seconds. */,
                             double             fps /**< [in] Measured camera sampling rate. */ );

  private:
    static constexpr size_t c_blockFrames = 128; ///< Short, contiguous window for the pixel periodogram.
    static constexpr size_t c_blocks      = 4;   ///< Sample windows distributed over the modal PSD interval.

    /// FFTW-deleter for aligned FFT work arrays.
    struct FftFree
    {
        /// Free FFTW work storage.
        void operator()( void *ptr /**< [in] Allocation to release. */ ) const
        {
            fftw_free( ptr );
        }
    };

    using Fft = mx::math::ft::fftT<float, std::complex<float>, 1, 0>; ///< Reusable camera temporal transform.
    Fft                                           m_fft;      ///< FFT plan, used only by the gain optimizer thread.
    std::unique_ptr<float, FftFree>               m_time;     ///< Aligned time-domain FFT work area.
    std::unique_ptr<std::complex<float>, FftFree> m_spectrum; ///< Aligned frequency-domain work area.
    std::vector<float>                            m_window;   ///< Periodogram Hann window.
};

inline Eigen::ArrayXd
CameraNoise::estimate( const std::string &name, const Calibration &calibration, double start, double end, double fps )
{
    char            filename[4096];
    std::error_code error;

    if( name.empty() || ImageStreamIO_filename( filename, sizeof( filename ), name.c_str() ) != IMAGESTREAMIO_SUCCESS ||
        !std::filesystem::exists( filename, error ) || end <= start || fps <= 0 )
    {
        return {};
    }

    IMAGE image{};

    if( ImageStreamIO_openIm( &image, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
    {
        return {};
    }

    std::unique_ptr<IMAGE, decltype( &ImageStreamIO_closeIm )> close( &image, ImageStreamIO_closeIm );
    return estimate( image, calibration, start, end, fps );
}

inline Eigen::ArrayXd
CameraNoise::estimate( const IMAGE &image, const Calibration &calibration, double start, double end, double fps )
{
    const auto *md = image.md;

    if( !md || !image.atimearray || !image.cntarray || !image.array.UI16 || !mx::math::isFinite( fps ) || fps <= 0 ||
        end <= start || md->naxis != 3 || ( md->imagetype & 0xF000F ) != ( CIRCULAR_BUFFER | ZAXIS_TEMPORAL ) ||
        md->datatype != _DATATYPE_UINT16 || !calibration.matchesFrame( md->size[0], md->size[1] ) )
    {
        return {};
    }

    const uint64_t depth       = md->size[2];
    const size_t   framePixels = static_cast<size_t>( md->size[0] ) * md->size[1];
    const auto    &pixels      = calibration.pixels();
    const uint64_t latest      = __atomic_load_n( &md->cnt1, __ATOMIC_ACQUIRE );

    if( latest >= depth || depth <= c_blockFrames + 2 )
    {
        return {};
    }

    const uint64_t latestCount = __atomic_load_n( &image.cntarray[latest], __ATOMIC_ACQUIRE );
    const double   latestTime  = seconds( image.atimearray[latest] );

    if( latestCount == 0 || latest != __atomic_load_n( &md->cnt1, __ATOMIC_ACQUIRE ) ||
        latestCount != __atomic_load_n( &md->cnt0, __ATOMIC_ACQUIRE ) )
    {
        return {};
    }

    if( !m_time )
    {
        m_window.resize( c_blockFrames );
        mx::sigproc::window::hann( m_window );
        m_time.reset( mx::math::ft::fftw_malloc<float>( c_blockFrames ) );
        m_spectrum.reset( mx::math::ft::fftw_malloc<std::complex<float>>( c_blockFrames / 2 + 1 ) );
        m_fft.plan( c_blockFrames, mx::math::ft::dir::forward, false );
    }

    constexpr size_t firstBin = ( 4 * ( c_blockFrames / 2 ) + 4 ) / 5;
    constexpr size_t endBin   = c_blockFrames / 2;
    const double     normalization =
        ( endBin - firstBin ) * Eigen::Map<const Eigen::ArrayXf>( m_window.data(), c_blockFrames ).square().sum();
    Eigen::Array<float, c_blockFrames, Eigen::Dynamic> frames( c_blockFrames, pixels.size() );
    Eigen::ArrayXd                                     power = Eigen::ArrayXd::Zero( pixels.size() );

    for( size_t block = 0; block < c_blocks; ++block )
    {
        const double center = start + ( end - start ) * ( block + 0.5 ) / c_blocks;
        const double age    = ( latestTime - center ) * fps + c_blockFrames / 2;

        // Leave the latest slot and the next write slot out of the snapshot.
        if( age < c_blockFrames || age >= depth - 1.5 || age >= latestCount - 0.5 )
        {
            return {};
        }

        const uint64_t firstAge = std::llround( age );
        double         previous = 0;

        for( size_t frame = 0; frame < c_blockFrames; ++frame )
        {
            const uint64_t offset = firstAge - frame;
            const uint64_t slot   = ( latest + depth - offset ) % depth;
            const double   time   = seconds( image.atimearray[slot] );

            if( __atomic_load_n( &image.cntarray[slot], __ATOMIC_ACQUIRE ) != latestCount - offset || time < start ||
                time > end || std::abs( ( time - center ) * fps + c_blockFrames / 2 - frame ) > 2.5 ||
                ( frame > 0 && std::abs( ( time - previous ) * fps - 1 ) > 0.25 ) )
            {
                return {};
            }

            const uint16_t *data = image.array.UI16 + framePixels * slot;

            for( size_t p = 0; p < pixels.size(); ++p )
            {
                frames( frame, p ) = data[pixels[p]];
            }

            previous = time;
        }

        // Reject a block if the producer could have started reusing its oldest frame.
        if( __atomic_load_n( &md->cnt0, __ATOMIC_ACQUIRE ) - ( latestCount - firstAge ) >= depth - 1 )
        {
            return {};
        }

        const Eigen::Map<const Eigen::ArrayXf> window( m_window.data(), c_blockFrames );
        Eigen::Map<Eigen::ArrayXf>             work( m_time.get(), c_blockFrames );

        for( size_t p = 0; p < pixels.size(); ++p )
        {
            work = ( frames.col( p ) - frames.col( p ).mean() ) * window;
            m_fft( m_spectrum.get(), m_time.get() );

            for( size_t f = firstBin; f < endBin; ++f )
            {
                power[p] += std::norm( m_spectrum.get()[f] );
            }
        }
    }

    return power / ( c_blocks * normalization );
}

/// Convert the recovered WFS disturbance and sampled detector noise to DM units.
inline void referToDm( std::vector<float>       &disturbance /**< [in,out] WFS disturbance PSD. */,
                       std::vector<float>       &noise /**< [in,out] Sampled detector-noise PSD. */,
                       const std::vector<float> &freq /**< [in] Frequency grid in Hz. */,
                       double                    fps /**< [in] Exposure cadence in Hz. */,
                       double                    responsePower /**< [in] Squared static WFS response. */,
                       std::initializer_list<std::vector<float> *> otherDisturbances =
                           {} /**< [in,out] Other WFS disturbance estimates. */ )
{
    for( size_t f = 0; f < freq.size(); ++f )
    {
        const double phase    = mx::math::pi<double>() * freq[f] / fps;
        const double exposure = phase == 0 ? 1 : std::sin( phase ) / phase;
        const double response = responsePower * exposure * exposure;
        disturbance[f] /= response;

        for( auto *other : otherDisturbances )
        {
            ( *other )[f] /= response;
        }

        noise[f] /= responsePower;
    }
}

} // namespace modalNoise
} // namespace app
} // namespace MagAOX

#endif // modalGainOpt_modalNoiseModel_hpp
