/** \file psfAcq.hpp
 * \brief The MagAO-X PSF Fitter application header
 *
 * \ingroup psfAcq_files
 */

#ifndef psfAcq_hpp
#define psfAcq_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <mx/math/fit/fitGaussian.hpp>
#include <mx/improc/imageFilters.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

/** \defgroup psfAcq
 * \brief The MagAO-X PSF fitter.
 *
 * <a href="../handbook/operating/software/apps/psfAcq.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup psfAcq_files
 * \ingroup psfAcq
 */

namespace MagAOX
{
namespace app
{

struct darkShmimT
{
    static std::string configSection()
    {
        return "darkShmim";
    };

    static std::string indiPrefix()
    {
        return "dark";
    };
};

struct Star
{
public:

    /// Monotonic identifier used as a stable tie-breaker across equal-brightness stars.
    std::size_t id{ 0 };

    /// Star row coordinate in image pixel space.
    float x{ 0 };

    /// Star column coordinate in image pixel space.
    float y{ 0 };

    /// Peak pixel value of the fitted star.
    float max{ 0 };

    /// FWHM returned by the Gaussian fit in pixels.
    float fwhm{ 0 };

    /// Derived seeing value in arcseconds.
    float seeing{ 0 };

    /// Consecutive frames where this star was not updated.
    int missedFrames{ 0 };

private:

    /// Owned INDI property for this star's exported fit values.
    std::unique_ptr<pcf::IndiProperty> m_prop;

public:
    /// Default constructor.
    Star() = default;

    /// Disable copy constructor because the star owns a unique INDI property instance.
    Star( const Star & ) = delete;

    /// Disable copy assignment because the star owns a unique INDI property instance.
    Star & operator=( const Star & ) = delete;

    /// Enable move constructor.
    Star( Star && ) noexcept = default;

    /// Enable move assignment.
    Star & operator=( Star && ) noexcept = default;

    /// Access the owned INDI property.
    pcf::IndiProperty & prop()
    {
        if( !m_prop )
        {
            throw std::runtime_error( "attempt to access nullptr prop" );
        }

        return *m_prop;
    }

    /// Check whether an INDI property has been allocated.
    bool hasProp() const
    {
        return static_cast<bool>( m_prop );
    }

    /// Allocate the star's INDI property if needed.
    void allocate()
    {
        if( !m_prop )
        {
            m_prop = std::make_unique<pcf::IndiProperty>();
        }
    }

    /// Release the star's INDI property.
    void deallocate()
    {
        m_prop.reset();
    }
};

/// The MagAO-X PSF Fitter
/**
 * \ingroup psfAcq
 */
class psfAcq : public MagAOXApp<true>,
               public dev::shmimMonitor<psfAcq>,
               public dev::shmimMonitor<psfAcq, darkShmimT>,
               public dev::telemeter<psfAcq>
{
    // Give the test harness access.
    friend class psfAcq_test;

    friend class dev::shmimMonitor<psfAcq>;
    friend class dev::shmimMonitor<psfAcq, darkShmimT>;
    friend class dev::telemeter<psfAcq>;

  public:
    // The base shmimMonitor type
    typedef dev::shmimMonitor<psfAcq> shmimMonitorT;

    typedef dev::shmimMonitor<psfAcq, darkShmimT> darkShmimMonitorT;

    // The base telemeter type
    typedef dev::telemeter<psfAcq> telemeterT;

    /// Floating point type in which to do all calculations.
    typedef float realT;

    /** \name app::dev Configurations
     *@{
     */
    ///@}

  protected:
    /** \name Configurable Parameters
     *@{
     */

    std::string m_fpsSource; ///< Device name for getting fps if time-based averaging is used.  This device should have
                             ///< *.fps.current.

    uint16_t m_fitCircBuffMaxLength{ 3600 }; ///< Maximum length of the latency measurement circular buffers
    float m_fitCircBuffMaxTime{ 5 };         ///< Maximum time of the latency meaurement circular buffers

    float m_fwhmGuess{ 4 };
    ///@}

    mx::improc::eigenImage<float> m_image;
    mx::improc::eigenImage<float> m_sm;

    mx::improc::eigenImage<float> m_dark;

    double m_acqQuitTime {0};
    double m_acqPauseTime{2};

    int m_current_acq_star {-1}; // active star index used for seeing output
    int m_temp_acq_star {-1};
    bool m_updated{ false };
    float m_x{ 0 };
    float m_y{ 0 };
    int m_max_loops{ 5 };        // default to detecting a max of 5 stars
    int m_zero_area{ 8 };        // default to zeroing an 8x8 pixel area around stars it finds
    float m_threshold = { 7.0 }; // how many sigma away from the mean you want to classify a detection, default to
                                 // 7sigma
    float m_fwhm_threshold = { 4.0 }; // minumum fwhm to consider something a star
    float m_max_fwhm = { 40.0 }; // max fwhm to consider a star

    std::vector<float> m_first_x_vals = {};
    std::vector<float> m_first_y_vals = {};
    /// Tracked stars with associated INDI properties.
    std::vector<Star> m_detectedStars;

    /// Monotonic id counter used to stamp new stars for stable ordering tie-breaks.
    std::size_t m_nextStarId{ 0 };

    float m_dx{ 0 };
    float m_dy{ 0 };

    double m_plate_scale = .0795336;
    int m_old_num_stars{ 0 };
    int m_num_stars{ 0 };
    float m_seeing{ 0 };
    int m_acquire_star{ -1 }; // Testing for user to select star
    int m_seeing_star { -1 }; // star index exposed for seeing source; auto-set to 0 when stars are present
    int m_x_center{};     // 'center' of image or hot spot
    int m_y_center{};

    float m_fps{ 0 };

    /// Last observed `flipacq.presetName.out` switch state.
    bool m_flipAcqOutWasOn{ false };

    /// True once `m_flipAcqOutWasOn` has been initialized from INDI.
    bool m_flipAcqOutStateValid{ false };

    /// Last successful loop-exit telemetry dump time in seconds.
    double m_lastLoopExitTelemTime{ 0 };

    /// Snapshot of one star's telemetry fields for deferred emission.
    struct starTelemSample
    {
        /// Star x position in pixels.
        float x_pos;

        /// Star y position in pixels.
        float y_pos;

        /// Peak pixel value.
        float m_pix;

        /// FWHM in pixels.
        float fwhm;

        /// Seeing estimate in arcseconds.
        float seeing;
    };

    /// Emit one telemetry record per star sample.
    int emitStarTelemetry( const std::vector<starTelemSample> &starTelemetryValues /**< [in] per-star telemetry samples to emit. */ );

    /// Remove one tracked star and its INDI property/callback registration.
    /** Caller must hold `m_indiMutex`.
     */
    void removeStar( size_t index /**< [in] index of the tracked star to remove. */ );

    /// Delete star INDI properties and reset tracked acquisition stars.
    /** Caller must hold `m_indiMutex`.
     */
    void resetAcq();

    /// Register one tracked star's read-only INDI property using a rank-based label.
    /** Caller must hold `m_indiMutex`.
     */
    void registerStarProperty( Star &star,                  /**< [in,out] tracked star to register. */
                               std::size_t rankIndex /**< [in] brightness rank label index, `star_<rankIndex>`. */ );

    /// Remove one tracked star's INDI property without erasing the star state.
    /** Caller must hold `m_indiMutex`.
     */
    void unregisterStarProperty( Star &star /**< [in,out] tracked star to unregister. */ );

    /// Sort tracked stars by brightness and ensure labels are `star_0`, `star_1`, ...
    /** Caller must hold `m_indiMutex`.
     */
    void relabelStarsByBrightness();

    // Working memory for poke fitting
    mx::math::fit::fitGaussian2Dsym<float> m_gfit;

  public:
    /// Default c'tor.
    psfAcq();

    /// D'tor, declared and defined for noexcept.
    ~psfAcq() noexcept;

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl(
        mx::app::appConfigurator &_config /**< [in] an application configuration from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for psfAcq.
    /**
     * \returns 0 on no critical error
     * \returns -1 on an error requiring shutdown
     */
    virtual int appLogic();

    /// Shutdown the app.
    /**
     *
     */
    virtual int appShutdown();

    // shmimMonitor interface:
    int allocate( const dev::shmimT & );

    int processImage( void *curr_src, const dev::shmimT & );

    // shmimMonitor interface for referenc:
    int allocate( const darkShmimT & );

    int processImage( void *curr_src, const darkShmimT & );

  protected:
    std::mutex m_imageMutex;

  protected:
    /** \name INDI
     * @{
     */

    pcf::IndiProperty m_indiP_num_stars;

    pcf::IndiProperty m_indiP_seeing;

    pcf::IndiProperty m_indiP_fpsSource;
    INDI_SETCALLBACK_DECL( psfAcq, m_indiP_fpsSource );

    /// Subscription to `flipacq.presetName` switch updates.
    pcf::IndiProperty m_indiP_flipAcqPresetName;
    INDI_SETCALLBACK_DECL( psfAcq, m_indiP_flipAcqPresetName );

    // For user to select star
    pcf::IndiProperty m_indiP_acquire_star;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_acquire_star );

    // For user to set what star they want to use to calc seeing
    pcf::IndiProperty m_indiP_seeing_star;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_seeing_star );

    // toggling
    pcf::IndiProperty m_indiP_restartAcq;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_restartAcq );

    pcf::IndiProperty m_indiP_recordSeeing;
    INDI_NEWCALLBACK_DECL( psfAcq, m_indiP_recordSeeing );

    ///@}

    /** \name Telemeter Interface
     *
     * @{
     */
    /// No-op scheduler hook; loop-exit logic emits telemetry directly.
    int checkRecordTimes();

    /// Record telemetry for all properties of each detected star.
    int recordTelem( const telem_psfacq *telemTag /**< [in] telemetry tag used for overload resolution. */ );

    ///@}
};

inline psfAcq::psfAcq() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    darkShmimMonitorT::m_getExistingFirst = true;
    return;
}

inline psfAcq::~psfAcq() noexcept
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );
        m_detectedStars.clear();
    }
}

inline void psfAcq::setupConfig()
{
    shmimMonitorT::setupConfig( config );
    darkShmimMonitorT::setupConfig( config );
    TELEMETER_SETUP_CONFIG( config );

    config.add(
        "fitter.fpsSource",
        "",
        "fitter.fpsSource",
        argType::Required,
        "fitter",
        "fpsSource",
        false,
        "string",
        "Device name for getting fps if time-based averaging is used.  This device should have *.fps.current." );
    config.add( "fitter.max_loops",
                "",
                "fitter.max_loops",
                argType::Required,
                "fitter",
                "max_loops",
                false,
                "int",
                "Setting the number of stars to detect in processImage function." );
    config.add( "fitter.zero_area",
                "",
                "fitter.zero_area",
                argType::Required,
                "fitter",
                "zero_area",
                false,
                "int",
                "Setting the pixel area to zero out after detecting stars in processImage function." );
    config.add( "fitter.threshold",
                "",
                "fitter.threshold",
                argType::Required,
                "fitter",
                "threshold",
                false,
                "float",
                "setting how many sigma away from the mean you want to classify a detection." );
    config.add( "fitter.fwhm_threshold",
                "",
                "fitter.fwhm_threshold",
                argType::Required,
                "fitter",
                "fwhm_threshold",
                false,
                "float",
                "minumum fwhm to consider something a star." );
    config.add( "acquisition.x_center",
                "",
                "acquisition.x_center",
                argType::Required,
                "acquisition",
                "x_center",
                false,
                "int",
                "X value for 'center' of image." );
    config.add( "acquisition.y_center",
                "",
                "acquisition.y_center",
                argType::Required,
                "acquisition",
                "y_center",
                false,
                "int",
                "Y value for 'center' of image." );
}

inline int psfAcq::loadConfigImpl( mx::app::appConfigurator &_config )
{
    shmimMonitorT::loadConfig( _config );
    darkShmimMonitorT::loadConfig( _config );
    TELEMETER_LOAD_CONFIG( _config );

    _config( m_fpsSource, "fitter.fpsSource" );
    _config( m_max_loops, "fitter.max_loops" ); // Max number of stars to detect in processImage
    _config( m_zero_area, "fitter.zero_area" ); // pixel area to zero out in processImage when a star is detected
    _config( m_threshold, "fitter.threshold" ); // how many sigma away from the mean you want to classify a detection
    _config( m_fwhm_threshold,
             "fitter.fwhm_threshold" ); // how many sigma away from the mean you want to classify a detection

    _config( m_x_center, "acquisition.x_center" );         // star number to acquire
    _config( m_y_center, "acquisition.y_center" );         // star number to acquire

    return 0;
}

inline void psfAcq::loadConfig()
{
    loadConfigImpl( config );
}

inline int psfAcq::appStartup()
{
    if( shmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( darkShmimMonitorT::appStartup() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    TELEMETER_APP_STARTUP;

    if( m_fpsSource != "" )
    {
        REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsSource, std::string( "fps" ) );
    }

    REG_INDI_SETPROP( m_indiP_flipAcqPresetName, "flipacq", "presetName" );

    // creating toggling to restart the acquisition
    createStandardIndiRequestSw( m_indiP_restartAcq, "restart_acq", "Restart Acquisition", "psfAcq");
    registerIndiPropertyNew( m_indiP_restartAcq, INDI_NEWCALLBACK(m_indiP_restartAcq) );

    // INDI prop for seeing
    createROIndiNumber(m_indiP_seeing, "seeing");
    m_indiP_seeing.add(pcf::IndiElement("current"));
    m_indiP_seeing["current"].setValue( m_seeing ); //m_seeing gets assigned the seeing values of the star that is acquired
    registerIndiPropertyReadOnly( m_indiP_seeing );

    // create toggling for recording seeing
    createStandardIndiToggleSw( m_indiP_recordSeeing, "record_seeing", "Record Seeing");
    m_indiP_recordSeeing["toggle"].set(pcf::IndiElement::On);
    if( registerIndiPropertyNew( m_indiP_recordSeeing, INDI_NEWCALLBACK(m_indiP_recordSeeing)) < 0)
    {
       log<software_error>({__FILE__,__LINE__});
       return -1;
    }

    // INDI prop for user to select star
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_acquire_star, "acquire_star", 0, 20, 1, "%d", "", "" );
    m_indiP_acquire_star["current"].setValue( m_acquire_star );
    m_indiP_acquire_star["target"].setValue( m_acquire_star );

    // INDI prop for user to select seeing star
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_seeing_star, "seeing_star", 0, 20, 1, "%d", "", "" );
    m_indiP_seeing_star["current"].setValue( m_seeing_star );
    m_indiP_seeing_star["target"].setValue( m_seeing_star );

    // number of stars INDI prop
    createROIndiNumber(m_indiP_num_stars, "num_stars");
    m_indiP_num_stars.add(pcf::IndiElement("current"));
    m_indiP_num_stars["current"].setValue( m_num_stars );
    registerIndiPropertyReadOnly( m_indiP_num_stars );

    state( stateCodes::OPERATING );

    return 0;
}

inline int psfAcq::appLogic()
{
    if( shmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    if( darkShmimMonitorT::appLogic() < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__ } );
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    shmimMonitorT::updateINDI();
    darkShmimMonitorT::updateINDI();

    updateIfChanged( m_indiP_num_stars, "current", m_num_stars );
    updateIfChanged( m_indiP_seeing, "current", m_seeing );
    updateIfChanged( m_indiP_seeing_star, "current", m_seeing_star );
    updateIfChanged( m_indiP_seeing_star, "target", m_seeing_star );

    for( size_t n = 0; n < m_detectedStars.size() ; ++n )
    {
        if( !m_detectedStars[n].hasProp() )
        {
            continue;
        }

        updateIfChanged( m_detectedStars[n].prop(), "x", m_detectedStars[n].x );
        updateIfChanged( m_detectedStars[n].prop(), "y", m_detectedStars[n].y );
        updateIfChanged( m_detectedStars[n].prop(), "peak", m_detectedStars[n].max );
        updateIfChanged( m_detectedStars[n].prop(), "fwhm", m_detectedStars[n].fwhm );
    }
    return 0;
}

inline int psfAcq::appShutdown()
{
    shmimMonitorT::appShutdown();
    darkShmimMonitorT::appShutdown();
    TELEMETER_APP_SHUTDOWN;

    return 0;
}

inline int psfAcq::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    m_image.resize( shmimMonitorT::m_width, shmimMonitorT::m_height );
    m_image.setZero();

    m_sm.resize( m_image.rows(), m_image.cols() );

    m_updated = false;
    return 0;
}

// Function to calculate Euclidean distance between two stars
inline float calculateDistance( float x1, float y1, float x2, float y2 )
{
    return std::sqrt( ( x2 - x1 ) * ( x2 - x1 ) + ( y2 - y1 ) * ( y2 - y1 ) );
}

inline int psfAcq::processImage( void *curr_src, const dev::shmimT &dummy )
{
    // Pause acquisition updates while telescope nudge is in progress.
    if( mx::sys::get_curr_time() - m_acqQuitTime < m_acqPauseTime )
    {
        return 0;
    }

    static_cast<void>( dummy );

    if( curr_src == nullptr )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "received null image pointer" } );
    }

    mx::improc::eigenImage<float> workingImage;
    {
        std::unique_lock<std::mutex> lock( m_imageMutex ); //mutex scope

        if( m_image.rows() <= 0 || m_image.cols() <= 0 )
        {
            return 0;
        }

        const std::size_t pixelCount =
            static_cast<std::size_t>( shmimMonitorT::m_width ) * static_cast<std::size_t>( shmimMonitorT::m_height );
        if( pixelCount == 0 )
        {
            return 0;
        }

        const float *src = static_cast<const float *>( curr_src );
        if( m_dark.rows() == m_image.rows() && m_dark.cols() == m_image.cols() )
        {
            for( std::size_t nn = 0; nn < pixelCount; ++nn )
            {
                m_image.data()[nn] = src[nn] - m_dark.data()[nn];
            }
        }
        else
        {
            for( std::size_t nn = 0; nn < pixelCount; ++nn )
            {
                m_image.data()[nn] = src[nn];
            }
        }

        workingImage = m_image;
    }

    const int imageRows = workingImage.rows();
    const int imageCols = workingImage.cols();
    if( imageRows <= 0 || imageCols <= 0 )
    {
        return 0;
    }

    if( m_zero_area <= 0 || imageRows < 2 * m_zero_area + 1 || imageCols < 2 * m_zero_area + 1 )
    {
        return 0;
    }

    int maxRow = 0;
    int maxCol = 0;
    float maxValue = workingImage.maxCoeff( &maxRow, &maxCol );

    const int noiseRows = std::min( imageRows, 32 );
    const int noiseCols = std::min( imageCols, 32 );
    eigenImage<float> noiseBlock = workingImage.block( 0, 0, noiseRows, noiseCols );
    float mean = noiseBlock.mean();
    float variance = ( noiseBlock.array() - mean ).square().sum() / noiseBlock.size();
    variance = std::max( variance, 0.0f );
    float stddev = std::sqrt( variance );
    if( !std::isfinite( stddev ) || stddev <= std::numeric_limits<float>::epsilon() )
    {
        return 0;
    }

    float zScore = ( maxValue - mean ) / stddev;
    if( !std::isfinite( zScore ) )
    {
        return 0;
    }

    float fwhm = m_fwhm_threshold + 1;
    int loopCount = 0;

    bool sendNudge = false;
    double nudgeXArcsec = 0;
    double nudgeYArcsec = 0;
    std::vector<starTelemSample> loopExitTelemSamples;
    const int maxTrackedStars = std::max( m_max_loops, 0 );

    { //mutex scope
        std::unique_lock<std::mutex> lock( m_indiMutex );

        std::vector<bool> starUpdatedThisFrame( m_detectedStars.size(), false );
        auto addStar = [&]( float starX, float starY, float peak, float starFwhm, float starSeeing ) {
            Star newStar;
            newStar.id = m_nextStarId++;
            newStar.x = starX;
            newStar.y = starY;
            newStar.max = peak;
            newStar.fwhm = starFwhm;
            newStar.seeing = starSeeing;
            m_detectedStars.push_back( std::move( newStar ) );
            starUpdatedThisFrame.push_back( true );
        };

        while( zScore > m_threshold && fwhm > m_fwhm_threshold && loopCount < m_max_loops )
        {
            ++loopCount;
            m_gfit.set_itmax( 1000 );

            maxRow = std::clamp( maxRow, m_zero_area, imageRows - m_zero_area );
            maxCol = std::clamp( maxCol, m_zero_area, imageCols - m_zero_area );

            eigenImage<float> subImage = workingImage.block(
                maxRow - m_zero_area,
                maxCol - m_zero_area,
                m_zero_area * 2,
                m_zero_area * 2 );
            m_gfit.setArray( subImage.data(), subImage.rows(), subImage.cols() );
            m_gfit.setGuess( 0, maxValue, m_zero_area, m_zero_area, mx::math::func::fwhm2sigma( m_fwhmGuess ) );
            m_gfit.fit();

            fwhm = m_gfit.fwhm();
            maxValue = m_gfit.G();
            if( !std::isfinite( fwhm ) || !std::isfinite( maxValue ) )
            {
                break;
            }

            float starSeeing = fwhm * m_plate_scale;
            m_x = ( maxRow - m_zero_area ) + m_gfit.x0();
            m_y = ( maxCol - m_zero_area ) + m_gfit.y0();
            if( !std::isfinite( m_x ) || !std::isfinite( m_y ) || !std::isfinite( starSeeing ) )
            {
                break;
            }

            m_first_x_vals.push_back( m_x );
            m_first_y_vals.push_back( m_y );

            int starRow = std::clamp( static_cast<int>( m_x ), m_zero_area, imageRows - m_zero_area );
            int starCol = std::clamp( static_cast<int>( m_y ), m_zero_area, imageCols - m_zero_area );

            // Only detections that satisfy configured thresholds may update or create tracked stars.
            const bool passesStarThresholds =
                ( zScore > m_threshold ) && ( fwhm > m_fwhm_threshold ) && ( fwhm < m_max_fwhm ) &&
                std::isfinite( maxValue );

            if( passesStarThresholds )
            {
                constexpr int thresholdDistance = 20;
                bool matchedKnownStar = false;

                for( std::size_t starIndex = 0; starIndex < m_detectedStars.size(); ++starIndex )
                {
                    Star &star = m_detectedStars[starIndex];
                    float dist = calculateDistance( star.x, star.y, starRow, starCol );
                    if( dist >= thresholdDistance )
                    {
                        continue;
                    }

                    star.x = m_x;
                    star.y = m_y;
                    star.max = maxValue;
                    star.fwhm = fwhm;
                    star.seeing = starSeeing;
                    starUpdatedThisFrame[starIndex] = true;
                    matchedKnownStar = true;
                    break;
                }

                if( !matchedKnownStar && static_cast<int>( m_detectedStars.size() ) < maxTrackedStars )
                {
                    addStar( m_x, m_y, maxValue, fwhm, starSeeing );
                }
            }

            for( int rr = starRow - m_zero_area; rr < starRow + m_zero_area; ++rr )
            {
                for( int cc = starCol - m_zero_area; cc < starCol + m_zero_area; ++cc )
                {
                    workingImage( rr, cc ) = 0;
                }
            }

            maxValue = workingImage.maxCoeff( &maxRow, &maxCol );
            zScore = ( maxValue - mean ) / stddev;
            if( !std::isfinite( zScore ) )
            {
                break;
            }
        }

        // Drop a tracked star as soon as it misses one full frame update.
        constexpr int maxMissedFrames = 1;
        for( std::size_t n = m_detectedStars.size(); n > 0; --n )
        {
            std::size_t starIndex = n - 1;

            if( starIndex < starUpdatedThisFrame.size() && starUpdatedThisFrame[starIndex] )
            {
                m_detectedStars[starIndex].missedFrames = 0;
                continue;
            }

            ++m_detectedStars[starIndex].missedFrames;
            if( m_detectedStars[starIndex].missedFrames >= maxMissedFrames )
            {
                removeStar( starIndex );
            }
        }

        relabelStarsByBrightness();

        const int starCount = static_cast<int>( m_detectedStars.size() );
        if( m_acquire_star >= starCount || m_acquire_star < -1 )
        {
            m_acquire_star = -1;
        }

        if( m_acquire_star >= 0 && m_acquire_star < starCount )
        {
            m_acqQuitTime = mx::sys::get_curr_time();
            m_temp_acq_star = m_acquire_star;

            int deltaX = static_cast<int>( m_detectedStars[m_acquire_star].x ) - m_x_center;
            int deltaY = static_cast<int>( m_detectedStars[m_acquire_star].y ) - m_y_center;

            // Negative signs move the scope opposite the current star offset.
            nudgeXArcsec = -1 * static_cast<double>( deltaY ) * m_plate_scale;
            nudgeYArcsec = -1 * static_cast<double>( deltaX ) * m_plate_scale;
            sendNudge = true;

            resetAcq();
            m_acquire_star = -1;
        }

        m_num_stars = static_cast<int>( m_detectedStars.size() );
        if( m_num_stars > 0 )
        {
            m_seeing_star = 0;
            m_current_acq_star = 0;
            m_seeing = m_detectedStars[0].seeing;
            if( !std::isfinite( m_seeing ) )
            {
                m_seeing = 0;
            }

            loopExitTelemSamples.reserve( m_detectedStars.size() );
            for( const auto &star : m_detectedStars )
            {
                if( !std::isfinite( star.x ) || !std::isfinite( star.y ) || !std::isfinite( star.max ) ||
                    !std::isfinite( star.fwhm ) || !std::isfinite( star.seeing ) )
                {
                    continue;
                }

                loopExitTelemSamples.push_back( { star.x, star.y, star.max, star.fwhm, star.seeing } );
            }
        }
        else
        {
            m_seeing = 0;
            m_seeing_star = -1;
            m_current_acq_star = -1;
        }

        m_updated = true;
    }

    if( !loopExitTelemSamples.empty() )
    {
        double now = mx::sys::get_curr_time();
        if( now - m_lastLoopExitTelemTime >= 1.0 )
        {
            if( emitStarTelemetry( loopExitTelemSamples ) < 0 )
            {
                return -1;
            }

            m_lastLoopExitTelemTime = now;
        }
    }

    if( sendNudge )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Number );
        ip.setDevice( "tcsi" );
        ip.setName( "pyrNudge" );
        ip.add( pcf::IndiElement( "y" ) );
        ip["y"] = nudgeXArcsec;
        ip.add( pcf::IndiElement( "x" ) );
        ip["x"] = nudgeYArcsec;
        sendNewProperty( ip );
    }

    return 0;
}

inline int psfAcq::allocate( const darkShmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> guard( m_imageMutex );

    if( darkShmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "dark is not float" } );
    }

    m_dark.resize( darkShmimMonitorT::m_width, darkShmimMonitorT::m_height );
    m_dark.setZero();

    return 0;
}

inline int psfAcq::processImage( void *curr_src, const darkShmimT &dummy )
{
    static_cast<void>( dummy );

    if( curr_src == nullptr )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "received null dark image pointer" } );
    }

    std::unique_lock<std::mutex> lock( m_imageMutex );
    const float *src = static_cast<const float *>( curr_src );

    for( unsigned nn = 0; nn < darkShmimMonitorT::m_width * darkShmimMonitorT::m_height; ++nn )
    {
        m_dark.data()[nn] += src[nn];
    }

    lock.unlock();

    log<text_log>( "dark updated", logPrio::LOG_INFO );

    return 0;
}

inline int psfAcq::checkRecordTimes()
{
    return 0;
}

inline int psfAcq::emitStarTelemetry( const std::vector<starTelemSample> &starTelemetryValues )
{
    if( starTelemetryValues.empty() )
    {
        return 0;
    }

    int numStars = static_cast<int>( starTelemetryValues.size() );
    for( std::size_t index = 0; index < starTelemetryValues.size(); ++index )
    {
        const auto &sample = starTelemetryValues[index];
        int starNo = static_cast<int>( index ) + 1;

        if( telem<telem_psfacq>(
                { starNo, numStars, sample.x_pos, sample.y_pos, sample.m_pix, sample.fwhm, sample.seeing } ) < 0 )
        {
            return -1;
        }
    }

    return 0;
}

inline int psfAcq::recordTelem( const telem_psfacq *telemTag )
{
    static_cast<void>( telemTag );

    std::vector<starTelemSample> starTelemetryValues;
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        if( m_detectedStars.empty() )
        {
            return 0;
        }

        starTelemetryValues.reserve( m_detectedStars.size() );
        for( const auto &star : m_detectedStars )
        {
            if( !std::isfinite( star.x ) || !std::isfinite( star.y ) || !std::isfinite( star.max ) ||
                !std::isfinite( star.fwhm ) || !std::isfinite( star.seeing ) )
            {
                continue;
            }

            starTelemetryValues.push_back( { star.x, star.y, star.max, star.fwhm, star.seeing } );
        }
    }

    return emitStarTelemetry( starTelemetryValues );
}

void psfAcq::registerStarProperty( Star &star, std::size_t rankIndex )
{
    // Caller must hold m_indiMutex.
    if( star.hasProp() )
    {
        unregisterStarProperty( star );
    }

    star.allocate();

    std::string starPrefix = "star_" + std::to_string( rankIndex );
    createROIndiNumber(
        star.prop(),
        starPrefix,
        "Star " + std::to_string( rankIndex ) + " Properties",
        "Star Acq" );
    star.prop().add( pcf::IndiElement( "x" ) );
    star.prop()["x"].set( star.x );
    star.prop().add( pcf::IndiElement( "y" ) );
    star.prop()["y"].set( star.y );
    star.prop().add( pcf::IndiElement( "peak" ) );
    star.prop()["peak"].set( star.max );
    star.prop().add( pcf::IndiElement( "fwhm" ) );
    star.prop()["fwhm"].set( star.fwhm );
    registerIndiPropertyReadOnly( star.prop() );
}

void psfAcq::unregisterStarProperty( Star &star )
{
    // Caller must hold m_indiMutex.
    if( !star.hasProp() )
    {
        return;
    }

    std::string uniqueKey = star.prop().createUniqueKey();
    if( m_indiDriver )
    {
        m_indiDriver->sendDelProperty( star.prop() );
    }

    if( !uniqueKey.empty() && !m_indiNewCallBacks.erase( uniqueKey ) )
    {
        log<software_error>( { __FILE__, __LINE__, "failed to erase " + uniqueKey } );
    }

    star.deallocate();
}

void psfAcq::relabelStarsByBrightness()
{
    // Caller must hold m_indiMutex.
    std::sort(
        m_detectedStars.begin(),
        m_detectedStars.end(),
        []( const Star &lhs, const Star &rhs ) {
            const bool lhsFinite = std::isfinite( lhs.max );
            const bool rhsFinite = std::isfinite( rhs.max );

            if( lhsFinite != rhsFinite )
            {
                return lhsFinite;
            }

            if( lhsFinite && rhsFinite && lhs.max != rhs.max )
            {
                return lhs.max > rhs.max;
            }

            return lhs.id < rhs.id;
        } );

    bool labelsAlreadyMatch = true;
    for( std::size_t n = 0; n < m_detectedStars.size(); ++n )
    {
        if( !m_detectedStars[n].hasProp() )
        {
            labelsAlreadyMatch = false;
            break;
        }

        std::string desiredName = "star_" + std::to_string( n );
        if( m_detectedStars[n].prop().getName() != desiredName )
        {
            labelsAlreadyMatch = false;
            break;
        }
    }

    if( labelsAlreadyMatch )
    {
        return;
    }

    for( auto &star : m_detectedStars )
    {
        unregisterStarProperty( star );
    }

    for( std::size_t n = 0; n < m_detectedStars.size(); ++n )
    {
        registerStarProperty( m_detectedStars[n], n );
    }
}

void psfAcq::removeStar( size_t index )
{
    // Caller must hold m_indiMutex.
    if( index >= m_detectedStars.size() )
    {
        return;
    }

    unregisterStarProperty( m_detectedStars[index] );
    m_detectedStars.erase( m_detectedStars.begin() + index );
}

// Delete INDI properties for tracked stars and clear acquisition state.
void psfAcq::resetAcq()
{
    // Caller must hold m_indiMutex.
    for( size_t n = m_detectedStars.size(); n > 0; --n )
    {
        removeStar( n - 1 );
    }

    m_num_stars = 0;
    m_seeing = 0;
    m_seeing_star = -1;
    m_current_acq_star = -1;
}

INDI_SETCALLBACK_DEFN( psfAcq, m_indiP_flipAcqPresetName )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_flipAcqPresetName, ipRecv );

    // Auto-restart on an `out` On->Off transition.
    // `flipacq.presetName` is a switch property with elements `in` and `out`.
    if( ipRecv.find( "out" ) != true )
    {
        return 0;
    }

    std::unique_lock<std::mutex> lock( m_indiMutex );

    auto outState = ipRecv["out"].getSwitchState();
    bool outIsOn = ( outState == pcf::IndiElement::On );
    bool outIsOff = ( outState == pcf::IndiElement::Off );

    if( !outIsOn && !outIsOff )
    {
        return 0;
    }

    if( m_flipAcqOutStateValid && m_flipAcqOutWasOn && outIsOff )
    {
        resetAcq();
    }

    m_flipAcqOutWasOn = outIsOn;
    m_flipAcqOutStateValid = true;

    return 0;
}

//for toggling Restart Acquisition
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_restartAcq )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_restartAcq, ipRecv);
    if(!ipRecv.find("request")) return 0;
    std::unique_lock<std::mutex> lock(m_indiMutex);

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On)
    {
        resetAcq();
        return 0;
    }
    else if( ipRecv["request"].getSwitchState() == pcf::IndiElement::Off)
    {
        return 0;
    }

    log<software_error>({__FILE__,__LINE__, "switch state fall through."});
    return -1;
}

// For toggling Recording Seeing
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_recordSeeing )(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_recordSeeing, ipRecv);
    if(!ipRecv.find("toggle")) return 0;
    std::unique_lock<std::mutex> lock(m_indiMutex);

    // Seeing updates are automatic; this toggle is informational only.
    if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
    {
        updateSwitchIfChanged(m_indiP_recordSeeing, "toggle", pcf::IndiElement::On, INDI_OK);
        return 0;
    }
    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off)
    {
        updateSwitchIfChanged(m_indiP_recordSeeing, "toggle", pcf::IndiElement::Off, INDI_IDLE);
        return 0;
    }


    log<software_error>({__FILE__,__LINE__, "switch state fall through."});
    return -1;
}

// For user to select acquisition star number
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_acquire_star )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_acquire_star.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_acquire_star, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );
        m_acquire_star = static_cast<int>( target );
    }

    log<text_log>( "set acquire_star = " + std::to_string( m_acquire_star ), logPrio::LOG_NOTICE );
    return 0;
}

// For user to select seeing star number
INDI_NEWCALLBACK_DEFN( psfAcq, m_indiP_seeing_star )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_seeing_star.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "wrong INDI property received." } );
        return -1;
    }

    float target;

    if( indiTargetUpdate( m_indiP_seeing_star, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    static_cast<void>( target );

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );

        // Force automatic seeing source selection.
        if( m_num_stars > 0 )
        {
            m_seeing_star = 0;
        }
        else
        {
            m_seeing_star = -1;
        }
    }

    log<text_log>( "set seeing_star = " + std::to_string( m_seeing_star ), logPrio::LOG_NOTICE );
    return 0;
}

INDI_SETCALLBACK_DEFN( psfAcq, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_fpsSource.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "Invalid INDI property." } );
        return -1;
    }

    if( ipRecv.find( "current" ) != true ) // this isn't valie
    {
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv["current"].get<float>();

    if( fps != m_fps )
    {
        m_fps = fps;
        shmimMonitorT::m_restart = true;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // psfAcq_hpp
