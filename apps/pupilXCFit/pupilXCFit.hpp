/** \file pupilXCFit.hpp
 * \brief Cross-correlate averaged pyramid pupils with a selected reference.
 * \ingroup pupilXCFit_files
 */

#ifndef pupilXCFit_hpp
#define pupilXCFit_hpp

#include <array>
#include <atomic>
#include <condition_variable>
#include <limits>
#include <optional>

#include <mx/improc/imageXCorrFFT.hpp>
#include <mx/math/floatUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

/** \defgroup pupilXCFit Pyramid pupil cross-correlation fitter
 * \ingroup apps
 */

/** \defgroup pupilXCFit_files
 * \ingroup pupilXCFit
 */

namespace MagAOX
{
namespace app
{

/// Monitor tag for either independently populated reference stream.
template <bool fitT>
struct pupilXCReferenceT
{
    /// Return the reference monitor's configuration section.
    static std::string configSection();

    /// Return the prefix of the reference monitor's INDI properties.
    static std::string indiPrefix();
};

/// Measure live-minus-reference pupil shifts in pixels.
/**
 * The input is an upstream dark-subtracted average. Each output is a 2 × 1 float frame with x followed by y.
 * An invalid measurement is represented by two NaNs. The output acquisition time is inherited from the input.
 * \ingroup pupilXCFit
 */
class pupilXCFit : public MagAOXApp<true>,
                   public dev::shmimMonitor<pupilXCFit>,
                   public dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<false>>,
                   public dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<true>>,
                   public dev::frameGrabber<pupilXCFit>,
                   public dev::telemeter<pupilXCFit>
{
    friend class dev::shmimMonitor<pupilXCFit>;
    friend class dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<false>>;
    friend class dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<true>>;
    friend class dev::frameGrabber<pupilXCFit>;
    friend class dev::telemeter<pupilXCFit>;

  protected:
    /// Float image in ImageStreamIO pixel order.
    using imageT = mx::improc::eigenImage<float>;

    /// Two-axis shift in stream order.
    using shiftT = std::array<float, 2>;

    /// Monitor for the live averaged image.
    using shmimMonitorT = dev::shmimMonitor<pupilXCFit>;

    /// Monitor for the normal WFS reference.
    using wfsRefMonitorT = dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<false>>;

    /// Monitor for the separately loaded fit reference.
    using fitRefMonitorT = dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<true>>;

    /// Output publisher for the two-axis measurement stream.
    using frameGrabberT = dev::frameGrabber<pupilXCFit>;

    /// Telemetry interface used for frame-grabber timing records.
    using telemeterT = dev::telemeter<pupilXCFit>;

    /// The measurement stream has a fixed axis ordering.
    static constexpr bool c_frameGrabber_flippable{ false };

    /// Marker for an unavailable correlation result or an invalid output sample.
    static constexpr float c_invalid{ std::numeric_limits<float>::quiet_NaN() };

    /// Correlation result before applying the peak and capture limits.
    struct fitResult
    {
        shiftT m_shift{ c_invalid, c_invalid }; ///< Live-minus-reference displacement in pixels.
        float  m_peak{ c_invalid };             ///< Correlation peak normalized by image size.
    };

    /// A sample and the calibration generation against which it was measured.
    struct sample
    {
        shiftT   m_shift;      ///< Measured x and y, or two NaNs on failure.
        timespec m_acquired;   ///< Acquisition time taken from the live monitor.
        uint64_t m_generation; ///< Reference generation when the sample was produced.
    };

    /** \name Fit Configuration - Data
     * @{
     */
    /// Largest accepted absolute displacement on either axis, in pixels.
    int m_maxShift{ 10 };

    /// Minimum normalized correlation peak for an accepted measurement.
    float m_minPeak{ 0.5f };
    ///@}

    /// Selected reference index, protected with the fit state by m_indiMutex.
    bool m_fitSelected{ false };

    /// Whether the correlator holds the selected reference at the live image size.
    bool m_referenceReady{ false };

    /// Reference pixels cached independently by the two monitors.
    std::array<imageT, 2> m_references;

    /// Reference snapshots remain usable until their monitor invalidates the attachment.
    std::array<std::atomic<bool>, 2> m_refPresent{};

    /// Whether the live monitor has allocated usable geometry.
    std::atomic<bool> m_livePresent{ false };

    /// Contiguous workspace holding a single private copy of the live frame.
    imageT m_live;

    /// FFT workspaces and the selected reference transform.
    mx::improc::imageXCorrFFT<imageT> m_xcorr;

    /// Incremented whenever a selected calibration is withdrawn or replaced.
    uint64_t m_generation{ 0 };

    /// Latest unpublished output, protected by m_indiMutex.
    std::optional<sample> m_pending;

    /// Sample claimed by the output worker before stream publication.
    sample m_claimed{};

    /// Wakes the output worker when a measurement or shutdown is pending.
    std::condition_variable m_outputReady;

    /// Whether shutdown has asked the worker to stop, protected by m_indiMutex.
    bool m_stopping{ false };

    /// INDI reference selection, whose state reports whether the calibration is usable.
    pcf::IndiProperty m_indiP_reference;

    /// Latest valid x and y shifts, with an alert state for rejected samples.
    pcf::IndiProperty m_indiP_average;

  public:
    /// Construct the fitter with the repository version.
    pupilXCFit();

    /// Destroy the application after its workers have stopped.
    ~pupilXCFit() noexcept override = default;

    /// Register application options and the stream interfaces.
    void setupConfig() override;

    /// Load and validate the application configuration.
    void loadConfig() override;

    /// Register INDI properties and start the stream interfaces.
    int appStartup() override;

    /// Monitor worker health and publish reference readiness.
    int appLogic() override;

    /// Wake and join the stream workers.
    int appShutdown() override;

  protected:
    /// Load application options from the supplied configuration.
    int loadConfigImpl( mx::app::appConfigurator &cfg /**< [in] application configuration */ );

    /// Apply a reference choice requested over INDI.
    INDI_NEWCALLBACK_DECL( pupilXCFit, m_indiP_reference );

    /// Allocate the live workspace for a new stream attachment.
    int allocate( const dev::shmimT &tag /**< [in] live monitor tag */ );

    /// Copy one live frame, measure its shift and queue a result.
    int processImage( void              *source /**< [in] live float image */,
                      const dev::shmimT &tag /**< [in] live monitor tag */ );

    /// Clear the previous reference when a stream attaches.
    template <bool fitT>
    int allocate( const pupilXCReferenceT<fitT> &tag /**< [in] reference monitor tag */ );

    /// Cache a reference image and invalidate the transform if it is selected.
    template <bool fitT>
    int processImage( void                          *source /**< [in] reference pixels */,
                      const pupilXCReferenceT<fitT> &tag /**< [in] reference monitor tag */ );

    /// Clear live availability on disconnect without locking the monitor hook.
    void shmimMonitorTiming( const dev::shmimT     &tag /**< [in] live monitor tag */,
                             dev::shmimMonitorEvent event /**< [in] monitor event */ );

    /// Clear reference availability on disconnect without locking the monitor hook.
    template <bool fitT>
    void shmimMonitorTiming( const pupilXCReferenceT<fitT> &tag /**< [in] reference monitor tag */,
                             dev::shmimMonitorEvent         event /**< [in] monitor event */ );

    /// Configure a float stream containing two values per frame.
    int configureAcquisition();

    /// Acknowledge output stream startup.
    int startAcquisition();

    /// Report unknown cadence for this event-driven publisher.
    float fps();

    /// Wait for and claim an output frame.
    int acquireAndCheckValid();

    /// Write the claimed measurement, substituting an invalid marker if its reference changed.
    int loadImageIntoStream( void *destination /**< [out] pointer to two floats */ );

    /// Acknowledge a frame-grabber restart.
    int reconfig();

    /// Publish frame-grabber timing records at their scheduled interval.
    int checkRecordTimes();

    /// Request a frame-grabber timing snapshot.
    int recordTelem( const telem_fgtimings *tag /**< [in] telemetry dispatch tag */ );

    /// Check that the image has finite pixels and nonzero, finite variance.
    static bool validSignal( const imageT &image /**< [in] candidate image for normalization */ );

    /// Construct or reuse the selected reference's transform while holding m_indiMutex.
    bool prepareReference();

    /// Measure one private live image while holding m_indiMutex.
    fitResult measure();

    /// Withdraw the selected calibration and replace any queued sample with an invalidation.
    void invalidateReference();

    /// Queue an output and update the INDI measurement while holding m_indiMutex.
    void publish( const std::optional<shiftT> &shift /**< [in] accepted shift, if available */,
                  const timespec              &acquired /**< [in] acquisition time associated with the result */ );

    /// Publish reference selection and readiness while holding m_indiMutex.
    void updateReferenceSelection();
};

template <bool fitT>
std::string pupilXCReferenceT<fitT>::configSection()
{
    return fitT ? "fitRefShmim" : "refShmim";
}

template <bool fitT>
std::string pupilXCReferenceT<fitT>::indiPrefix()
{
    return fitT ? "fitRef" : "ref";
}

inline pupilXCFit::pupilXCFit() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
}

inline void pupilXCFit::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    SHMIMMONITORT_SETUP_CONFIG( wfsRefMonitorT, config );
    SHMIMMONITORT_SETUP_CONFIG( fitRefMonitorT, config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );

    using mx::app::argType;
    config.add( "fit.reference",
                "",
                "fit.reference",
                argType::Required,
                "fit",
                "reference",
                false,
                "string",
                "Startup reference selection: wfs or fit (default wfs)." );
    config.add( "fit.maxShift",
                "",
                "fit.maxShift",
                argType::Required,
                "fit",
                "maxShift",
                false,
                "int",
                "Maximum accepted shift per axis in pixels (default 10)." );
    config.add( "fit.minPeak",
                "",
                "fit.minPeak",
                argType::Required,
                "fit",
                "minPeak",
                false,
                "float",
                "Minimum normalized correlation peak (default 0.5)." );
}

inline int pupilXCFit::loadConfigImpl( mx::app::appConfigurator &cfg )
{
    shmimMonitorT::m_shmimName         = "camwfs_avg";
    wfsRefMonitorT::m_shmimName        = "camwfs_ref";
    fitRefMonitorT::m_shmimName        = "camwfs_fit_ref";
    frameGrabberT::m_shmimName         = "camwfs_xc_fitdelta";
    wfsRefMonitorT::m_getExistingFirst = true;
    fitRefMonitorT::m_getExistingFirst = true;
    frameGrabberT::m_staged            = true;
    SHMIMMONITOR_LOAD_CONFIG( cfg );
    SHMIMMONITORT_LOAD_CONFIG( wfsRefMonitorT, cfg );
    SHMIMMONITORT_LOAD_CONFIG( fitRefMonitorT, cfg );
    FRAMEGRABBER_LOAD_CONFIG( cfg );
    TELEMETER_LOAD_CONFIG( cfg );

    std::string selection{ "wfs" };
    cfg( selection, "fit.reference" );
    cfg( m_maxShift, "fit.maxShift" );
    cfg( m_minPeak, "fit.minPeak" );

    if( ( selection != "wfs" && selection != "fit" ) || m_maxShift < 1 || !mx::math::isFinite( m_minPeak ) ||
        m_minPeak < 0 || m_minPeak > 1 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "invalid pupilXCFit fit configuration" } );
    }

    m_fitSelected = selection == "fit";
    return 0;
}

inline void pupilXCFit::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

inline int pupilXCFit::appStartup()
{
    createStandardIndiSelectionSw( m_indiP_reference, "reference", { "wfs", "fit" }, "Reference" );
    m_indiP_reference[m_fitSelected ? "fit" : "wfs"].setSwitchState( pcf::IndiElement::On );
    m_indiP_reference.setState( INDI_BUSY );

    if( registerIndiPropertyNew( m_indiP_reference, INDI_NEWCALLBACK( m_indiP_reference ) ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "registering reference selector failed" } );
    }

    createROIndiNumber( m_indiP_average, "average", "Pupil Shift" );
    indi::addNumberElement<float>( m_indiP_average, "dx", -m_maxShift, m_maxShift, 0, "%0.2f", "X shift (px)" );
    indi::addNumberElement<float>( m_indiP_average, "dy", -m_maxShift, m_maxShift, 0, "%0.2f", "Y shift (px)" );
    m_indiP_average.setState( INDI_BUSY );

    if( registerIndiPropertyReadOnly( m_indiP_average ) < 0 )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "registering measured shifts failed" } );
    }

    SHMIMMONITOR_APP_STARTUP;
    SHMIMMONITORT_APP_STARTUP( wfsRefMonitorT );
    SHMIMMONITORT_APP_STARTUP( fitRefMonitorT );
    TELEMETER_APP_STARTUP;
    FRAMEGRABBER_APP_STARTUP;
    state( stateCodes::OPERATING );
    return 0;
}

inline int pupilXCFit::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    SHMIMMONITORT_APP_LOGIC( wfsRefMonitorT );
    SHMIMMONITORT_APP_LOGIC( fitRefMonitorT );
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( m_referenceReady && ( !m_livePresent || !m_refPresent[m_fitSelected] ) )
    {
        invalidateReference();
    }

    SHMIMMONITOR_UPDATE_INDI;
    SHMIMMONITORT_UPDATE_INDI( wfsRefMonitorT );
    SHMIMMONITORT_UPDATE_INDI( fitRefMonitorT );
    FRAMEGRABBER_UPDATE_INDI;
    return 0;
}

inline int pupilXCFit::appShutdown()
{
    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );
        m_stopping = true;
    }

    m_outputReady.notify_one();
    SHMIMMONITOR_APP_SHUTDOWN;
    SHMIMMONITORT_APP_SHUTDOWN( wfsRefMonitorT );
    SHMIMMONITORT_APP_SHUTDOWN( fitRefMonitorT );
    FRAMEGRABBER_APP_SHUTDOWN;
    TELEMETER_APP_SHUTDOWN;
    return 0;
}

INDI_NEWCALLBACK_DEFN( pupilXCFit, m_indiP_reference )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reference, ipRecv );
    const bool wfs = ipRecv.find( "wfs" ) && ipRecv["wfs"].getSwitchState() == pcf::IndiElement::On;
    const bool fit = ipRecv.find( "fit" ) && ipRecv["fit"].getSwitchState() == pcf::IndiElement::On;

    if( wfs == fit )
    {
        return -1;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( fit != m_fitSelected )
    {
        m_fitSelected = fit;
        invalidateReference();
    }

    return 0;
}

inline int pupilXCFit::allocate( const dev::shmimT & )
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    using monitorT         = shmimMonitorT;
    const auto smallerSide = std::min( monitorT::m_width, monitorT::m_height );
    m_livePresent =
        monitorT::m_dataType == _DATATYPE_FLOAT && smallerSide / 6 > static_cast<unsigned>( m_maxShift ) + 1;
    invalidateReference();

    if( m_livePresent )
    {
        m_live.resize( monitorT::m_width, monitorT::m_height );
    }

    return 0;
}

inline int pupilXCFit::processImage( void *source, const dev::shmimT & )
{
    const uint64_t        frame    = shmimMonitorT::currCnt0();
    const timespec        acquired = shmimMonitorT::currAtime();
    fitResult             result;
    std::optional<shiftT> accepted;
    bool                  fitSelected;

    { //mutex scope
        std::lock_guard<std::mutex> guard( m_indiMutex );
        fitSelected = m_fitSelected;

        if( m_livePresent && prepareReference() )
        {
            std::copy_n( static_cast<const float *>( source ), m_live.size(), m_live.data() );
            result = measure();

            if( mx::math::isFinite( result.m_peak ) && result.m_peak >= m_minPeak &&
                std::abs( result.m_shift[0] ) <= m_maxShift && std::abs( result.m_shift[1] ) <= m_maxShift )
            {
                accepted = result.m_shift;
            }
        }

        publish( accepted, acquired );
    }

    const auto &reference = fitSelected ? fitRefMonitorT::m_shmimName : wfsRefMonitorT::m_shmimName;
    return telem<telem_pupilxcfit>(
        { frame, acquired, reference, accepted.has_value(), result.m_shift[0], result.m_shift[1], result.m_peak } );
}

template <bool fitT>
int pupilXCFit::allocate( const pupilXCReferenceT<fitT> & )
{
    m_refPresent[fitT] = false;
    std::lock_guard<std::mutex> guard( m_indiMutex );

    if( m_fitSelected == fitT )
    {
        invalidateReference();
    }

    return 0;
}

template <bool fitT>
int pupilXCFit::processImage( void *source, const pupilXCReferenceT<fitT> & )
{
    using monitorT = dev::shmimMonitor<pupilXCFit, pupilXCReferenceT<fitT>>;
    std::lock_guard<std::mutex> guard( m_indiMutex );
    auto                       &reference = m_references[fitT];
    m_refPresent[fitT]                    = false;

    if( monitorT::m_dataType == _DATATYPE_FLOAT && monitorT::m_width > 0 && monitorT::m_height > 0 )
    {
        reference.resize( monitorT::m_width, monitorT::m_height );
        std::copy_n( static_cast<const float *>( source ), reference.size(), reference.data() );
        m_refPresent[fitT] = validSignal( reference );
    }

    if( m_fitSelected == fitT )
    {
        invalidateReference();
    }

    return 0;
}

inline void pupilXCFit::shmimMonitorTiming( const dev::shmimT &, dev::shmimMonitorEvent event )
{
    if( event == dev::shmimMonitorEvent::disconnected )
    {
        m_livePresent = false;
    }
}

template <bool fitT>
void pupilXCFit::shmimMonitorTiming( const pupilXCReferenceT<fitT> &, dev::shmimMonitorEvent event )
{
    if( event == dev::shmimMonitorEvent::disconnected )
    {
        m_refPresent[fitT] = false;
    }
}

inline int pupilXCFit::configureAcquisition()
{
    frameGrabberT::m_width    = 2;
    frameGrabberT::m_height   = 1;
    frameGrabberT::m_dataType = _DATATYPE_FLOAT;
    return 0;
}

inline int pupilXCFit::startAcquisition()
{
    return 0;
}

inline float pupilXCFit::fps()
{
    return 0;
}

inline int pupilXCFit::reconfig()
{
    return 0;
}

inline int pupilXCFit::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int pupilXCFit::recordTelem( const telem_fgtimings * )
{
    return frameGrabberT::recordFGTimings( true );
}

inline int pupilXCFit::acquireAndCheckValid()
{
    std::unique_lock<std::mutex> lock( m_indiMutex );
    m_outputReady.wait_for( lock,
                            std::chrono::seconds( 1 ),
                            [this]
                            {
                                return m_stopping || m_pending.has_value();
                            } );

    if( m_stopping || !m_pending )
    {
        return 1;
    }

    m_claimed = *m_pending;
    m_pending.reset();
    frameGrabberT::m_currImageTimestamp = m_claimed.m_acquired;
    return 0;
}

inline int pupilXCFit::loadImageIntoStream( void *destination )
{
    std::lock_guard<std::mutex> guard( m_indiMutex );
    const bool current = m_claimed.m_generation == m_generation && m_livePresent && m_refPresent[m_fitSelected];
    const auto values  = current ? m_claimed.m_shift : shiftT{ c_invalid, c_invalid };
    std::copy( values.begin(), values.end(), static_cast<float *>( destination ) );
    return 0;
}

inline bool pupilXCFit::validSignal( const imageT &image )
{
    if( image.size() == 0 || !std::all_of( image.data(),
                                           image.data() + image.size(),
                                           []( float pixel )
                                           {
                                               return mx::math::isFinite( pixel );
                                           } ) )
    {
        return false;
    }

    const float variance = ( image - image.mean() ).square().mean();
    return mx::math::isFinite( variance ) && variance > 0;
}

inline bool pupilXCFit::prepareReference()
{
    const auto &reference = m_references[m_fitSelected];

    if( !m_refPresent[m_fitSelected] || reference.rows() != m_live.rows() || reference.cols() != m_live.cols() )
    {
        if( m_referenceReady )
        {
            invalidateReference();
        }

        return false;
    }

    if( m_referenceReady )
    {
        return true;
    }

    // mxlib's MFT supports square images. Interpolate the local FFT peak for rectangular images.
    const bool square = reference.rows() == reference.cols();
    m_xcorr.peakMethod( square ? mx::improc::xcorrPeakMethod::mftOversamp : mx::improc::xcorrPeakMethod::interpPeak );
    m_xcorr.mftLimitPeakSearch( square );
    m_xcorr.maxLag( m_maxShift + 1 );
    // The fixed MFT grid needs to cover both sides of the nearest integer-pixel peak.
    m_xcorr.tol( square ? std::max( 0.01f, 1.1f / reference.rows() ) : 0.02f );

    try
    {
        m_referenceReady = m_xcorr.refIm( reference, 1 ) == 0;
    }
    catch( const std::exception &error )
    {
        log<software_error>( { __FILE__, __LINE__, error.what() } );
    }

    updateReferenceSelection();
    return m_referenceReady;
}

inline pupilXCFit::fitResult pupilXCFit::measure()
{
    if( !validSignal( m_live ) )
    {
        return {};
    }

    fitResult result;

    try
    {
        if( m_xcorr( result.m_shift[0], result.m_shift[1], result.m_peak, m_live ) != 0 )
        {
            return {};
        }
    }
    catch( const std::exception &error )
    {
        log<software_error>( { __FILE__, __LINE__, error.what() } );
        return {};
    }

    if( !mx::math::isFinite( result.m_peak ) || !mx::math::isFinite( result.m_shift[0] ) ||
        !mx::math::isFinite( result.m_shift[1] ) )
    {
        return {};
    }

    result.m_peak /= m_live.size();
    return result;
}

inline void pupilXCFit::invalidateReference()
{
    timespec invalidated{};
    clock_gettime( CLOCK_REALTIME, &invalidated );
    m_referenceReady = false;
    ++m_generation;
    publish( {}, invalidated );
    updateReferenceSelection();
}

inline void pupilXCFit::publish( const std::optional<shiftT> &shift, const timespec &acquired )
{
    m_pending = sample{ shift.value_or( shiftT{ c_invalid, c_invalid } ), acquired, m_generation };
    m_outputReady.notify_one();

    if( shift )
    {
        updateIfChanged( m_indiP_average,
                         std::vector<std::string>{ "dx", "dy" },
                         std::vector<float>{ ( *shift )[0], ( *shift )[1] },
                         INDI_OK );
    }
    else if( m_indiDriver && m_indiP_average.getState() != INDI_ALERT )
    {
        m_indiP_average.setState( INDI_ALERT );
        m_indiDriver->sendSetProperty( m_indiP_average );
    }
}

inline void pupilXCFit::updateReferenceSelection()
{
    indi::updateSelectionSwitchIfChanged(
        m_indiP_reference, m_fitSelected ? "fit" : "wfs", m_indiDriver, m_referenceReady ? INDI_OK : INDI_BUSY );
}

} // namespace app
} // namespace MagAOX

#endif // pupilXCFit_hpp
