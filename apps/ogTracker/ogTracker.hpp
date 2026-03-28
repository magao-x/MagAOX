/** \file ogTracker.hpp
 * \brief Rolling PCA tracker for sparkle operating point validation.
 *
 * \ingroup ogTracker_files
 */

#ifndef ogTracker_hpp
#define ogTracker_hpp

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <mx/sigproc/circularBuffer.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#include "../../magaox_git_version.h"

namespace MagAOX
{
namespace app
{

/** \defgroup ogTracker ogTracker
 * \brief Rolling PCA tracker for sparkle operating-point validation.
 *
 * \ingroup apps
 */

/** \defgroup ogTracker_files ogTracker Files
 * \ingroup ogTracker
 */

/// MagAO-X app to project imWFS frames onto sparkle PCA references.
/**
 * \ingroup ogTracker
 */
class ogTracker :
    public MagAOXApp<true>,
    public dev::shmimMonitor<ogTracker>,
    public dev::telemeter<ogTracker>
{
    typedef float realT;

    friend class ogTracker_test;
    friend class dev::shmimMonitor<ogTracker>;
    friend class dev::telemeter<ogTracker>;

  public:
    typedef int32_t cbIndexT;
    typedef dev::shmimMonitor<ogTracker> shmimMonitorT;
    typedef dev::telemeter<ogTracker>    telemeterT;
    typedef mx::sigproc::circularBufferIndex<realT *, cbIndexT> frameCircBuffT;

  protected:
    /** \name Configuration - Data
     * @{
     */
    std::string m_calibRoot{ calibDir() + "/sparkPCA" }; ///< Root directory holding sparkle PCA calibrations.
    const std::string m_tweeterDevice{ "tweeterSpeck" }; ///< Fixed INDI device name for sparkle parameters.
    int         m_bufferN{ 2000 }; ///< Rolling frame-buffer length used for PCA statistics.
    int         m_minSamples{ 100 }; ///< Minimum buffered frames before statistics are considered valid.
    int         m_klipMax{ 3 }; ///< Maximum number of PCA modes to load/publish.
    int         m_ogAvgN{ 100 }; ///< Number of measurements used in the running average of `pca_og`.
    ///@}

    /** \name Sparkle Parameter State - Data
     * @{
     */
    float m_sep{ 0.0f }; ///< Current sparkle separation from INDI.
    float m_ang{ 0.0f }; ///< Current sparkle angle from INDI.
    float m_amp{ 0.0f }; ///< Current sparkle amplitude from INDI.
    float m_freq{ 0.0f }; ///< Current sparkle modulation frequency from INDI.
    bool  m_modulating{ false }; ///< True when tweeterSpeck modulation is enabled.
    ///@}

    /** \name Calibration State - Data
     * @{
     */
    std::string m_calibFolder; ///< Exact-match calibration folder name derived from sparkle parameters.
    std::string m_calibPath; ///< Full calibration folder path resolved from sparkle parameters.
    std::string m_calibError; ///< Human-readable calibration state/error string published to INDI.
    bool        m_calibLoaded{ false }; ///< True when reference PCA/RMS files are loaded and dimensionally valid.
    bool        m_paramsDirty{ true }; ///< Set when sparkle parameters change and calibration must be re-resolved.
    bool        m_waitForParamChange{ false }; ///< Hold retries after missing folder until a parameter changes.
    ///@}

    /** \name Frame Geometry and External Circular Buffer - Data
     * @{
     */
    int m_frameWidth{ 0 }; ///< Width of input frames from the monitored stream.
    int m_frameHeight{ 0 }; ///< Height of input frames from the monitored stream.
    int m_framePixels{ 0 }; ///< Cached flattened frame size (`width * height`).

    frameCircBuffT m_frameCircBuff; ///< Pointer circular buffer storing incoming `curr_src` frame pointers.
    int            m_bufferCapacity{ 0 }; ///< Effective capacity used by `m_frameCircBuff` after depth/config bounds.
    ///@}

    /** \name PCA Products and Outputs - Data
     * @{
     */
    Eigen::Matrix<realT, -1, -1> m_refPca; ///< Reference PCA basis matrix with shape `[pixels, modes]`.
    Eigen::Matrix<realT, -1, 1>  m_refRms; ///< Reference RMS vector with shape `[modes]`.
    int                           m_activeModes{ 0 }; ///< Number of active PCA modes used after file loading/cropping.

    Eigen::Matrix<realT, -1, 1> m_latestRms; ///< Latest rolling RMS per PCA mode.
    Eigen::Matrix<realT, -1, 1> m_latestNorm; ///< Latest rolling RMS normalized by reference RMS.
    Eigen::Matrix<realT, -1, 1> m_latestOgAvg; ///< Running-average output values for `pca_og`.
    realT                        m_latestOgSummary{ 0 }; ///< Mean of `pca_og_avg` modes using only finite values `<= 1`.
    bool                         m_metricsValid{ false }; ///< True once at least one valid metrics computation completes.
    Eigen::Matrix<realT, -1, -1> m_ogAvgHistory; ///< History matrix for running-average updates, shaped `[ogAvgN, modes]`.
    Eigen::Matrix<realT, -1, 1>  m_ogAvgSum; ///< Running sum across `m_ogAvgHistory`.
    int                           m_ogAvgWrite{ 0 }; ///< Next row index to overwrite in `m_ogAvgHistory`.
    int                           m_ogAvgCount{ 0 }; ///< Number of valid rows currently accumulated in `m_ogAvgHistory`.

    std::vector<std::string> m_modeEls; ///< Cached INDI element names (`mode0`, `mode1`, ...).
    ///@}

    /** \name INDI Properties - Data
     * @{
     */
    pcf::IndiProperty m_indiP_sep; ///< Subscription handle for sparkle separation.
    pcf::IndiProperty m_indiP_ang; ///< Subscription handle for sparkle angle.
    pcf::IndiProperty m_indiP_amp; ///< Subscription handle for sparkle amplitude.
    pcf::IndiProperty m_indiP_freq; ///< Subscription handle for sparkle frequency.
    pcf::IndiProperty m_indiP_modulating; ///< Subscription handle for sparkle modulation toggle.

    pcf::IndiProperty m_indiP_calibFolder; ///< Published calibration-folder property.
    pcf::IndiProperty m_indiP_calibError; ///< Published calibration-error property.
    pcf::IndiProperty m_indiP_calibLoaded; ///< Published calibration-loaded numeric flag.
    pcf::IndiProperty m_indiP_buffer; ///< Published ring-buffer occupancy/capacity property.
    pcf::IndiProperty m_indiP_ogAvgN; ///< INDI control for `pca_og` running-average measurement count.
    pcf::IndiProperty m_indiP_pcaOG; ///< Published instantaneous normalized rolling RMS values by PCA mode.
    pcf::IndiProperty m_indiP_pcaOGAvg; ///< Published running-average normalized values by PCA mode.
    pcf::IndiProperty m_indiP_pcaOGSummary; ///< Published scalar summary of `pca_og_avg` across modes.
    ///@}

    /** \name Concurrency and Background Compute - Data
     * @{
     */
    std::mutex m_dataMutex; ///< Protects shared state used by callbacks, app logic, and compute thread.
    std::condition_variable m_computeCv; ///< Wakes compute thread when new work is pending.
    std::thread             m_computeThread; ///< Worker thread performing PCA statistics on snapshots.
    bool                    m_computeRun{ false }; ///< Thread run flag, cleared during shutdown.
    bool                    m_computePending{ false }; ///< Set when a new compute pass should run.
    bool                    m_streamMissingLogged{ false }; ///< Debounce flag for missing/connected stream log transitions.
    bool                    m_waitModulationLogged{ false }; ///< Debounce flag for waiting/resume modulation logs.
    ///@}

  public:
    /// Default c'tor.
    ogTracker();
    /// D'tor, declared and defined for noexcept.
    ~ogTracker() noexcept
    {
    }
    /// => APPLICATION INTERFACE <= ///
    /// Set up configurable parameters and shmimMonitor options.
    virtual void setupConfig();

    /// Load configured values from appConfigurator.
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] application configuration source */ );

    /// Wrapper that invokes `loadConfigImpl(config)`.
    virtual void loadConfig();

    /// Register INDI properties, start monitors, and launch worker thread.
    virtual int appStartup();

    /// Main FSM logic: monitor stream state, handle calibration updates, publish outputs.
    virtual int appLogic();

    /// Stop worker thread and shut down shmim monitoring.
    virtual int appShutdown();

    /// => SHMIMMON INTERFACE <= ///
    int allocate( const dev::shmimT & );
    /// Consume one frame from the monitored stream and append its pointer to the external circular buffer.
    int processImage( void *curr_src /**< [in] pointer to current frame pixel data */,
                      const dev::shmimT & /**< [in] shmimMonitor tag to disambiguate overloads */ );

    /// => OGTRACKER INTERNAL FUNCTIONS <= ///
    /// Format sparkle parameters into the exact calibration-folder naming convention.
    static std::string formatCalibFolder( float sep /**< [in] sparkle separation */,
                                          float ang /**< [in] sparkle angle */,
                                          float amp /**< [in] sparkle amplitude */,
                                          float freq /**< [in] sparkle modulation frequency */ );

    /// Compute oldest-frame index for a latest-ended circular-buffer window.
    static int cbWindowStartIndex( int latestIndex /**< [in] index of most recent sample */,
                                   int count /**< [in] number of samples to include */,
                                   int size /**< [in] circular-buffer size used for wrapping */ );

    /// Compute RMS of each PCA-mode column in a projection matrix.
    static Eigen::Matrix<realT, -1, 1>
    rmsPerMode( const Eigen::Matrix<realT, -1, -1> &projection /**< [in] projection matrix shaped `[frames, modes]` */ );

    /// Normalize per-mode RMS values by reference RMS values.
    static Eigen::Matrix<realT, -1, 1>
    normalizeByReference( const Eigen::Matrix<realT, -1, 1> &rmsVals /**< [in] current rolling RMS values */,
                          const Eigen::Matrix<realT, -1, 1> &refVals /**< [in] reference RMS values from calibration */,
                          realT eps /**< [in] minimum absolute divisor threshold */ );

  protected:
    /// Update calibration status message and emit transition log while mutex is held.
    void setCalibErrorLocked( const std::string &msg /**< [in] new calibration-status string */,
                              logPrioT            prio = logPrio::LOG_WARNING /**< [in] log priority for state change */ );

    /// Resolve sparkle parameters to a folder and attempt calibration-file load.
    int  refreshCalibration();

    /// Load reference PCA and RMS files from a resolved calibration folder.
    int loadCalibrationFiles( const std::filesystem::path &folderPath /**< [in] fully-qualified calibration folder */ );

    /// Allocate/reset external pointer-cbuffer storage while mutex is held.
    int setupFrameCircBuffLocked( int pixels /**< [in] flattened frame size */ );

    /// Allocate/reset `pca_og` running-average history while mutex is held.
    int setupOgAverageLocked( int modes /**< [in] number of active PCA modes */ );

    /// Worker that computes metrics from ring snapshots.
    void computeThreadExec();

    /// Builds snapshot under lock then computes outside lock.
    void computeMetricsFromSnapshot();

    /// Telemetry periodic-record callback.
    int checkRecordTimes();

    /// Telemetry dispatch from telemeter.
    int recordTelem( const telem_dmspeck * );
    int recordTelem( const telem_dmmodes * );

    /// Emit sparkle/calibration-state telemetry.
    int recordDmSpeck( bool force = false );

    /// Emit og summary + mode-average telemetry.
    int recordOgModes( bool force = false );

  public:
    /// INDI properties we're tracking  
    INDI_SETCALLBACK_DECL( ogTracker, m_indiP_sep );
    INDI_SETCALLBACK_DECL( ogTracker, m_indiP_ang );
    INDI_SETCALLBACK_DECL( ogTracker, m_indiP_amp );
    INDI_SETCALLBACK_DECL( ogTracker, m_indiP_freq );
    INDI_SETCALLBACK_DECL( ogTracker, m_indiP_modulating );
    INDI_NEWCALLBACK_DECL( ogTracker, m_indiP_ogAvgN );
};

inline ogTracker::ogTracker() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    shmimMonitorT::m_shmimName = "aol1_imWFS2_cbuff";
    /// Make sure the image stream is running 
    shmimMonitorT::m_getExistingFirst = true;
    /// TODO: make sure that sparkles are running?
}

inline std::string ogTracker::formatCalibFolder( float sep, float ang, float amp, float freq )
{
    char folder[256];
    std::snprintf( folder,
                   sizeof( folder ),
                   "sep%02d_ang%02d_amp%01.3f_freq%02d",
                   static_cast<int>( sep ),
                   static_cast<int>( ang ),
                   amp,
                   static_cast<int>( freq ) );
    return folder;
}

inline int ogTracker::cbWindowStartIndex( int latestIndex, int count, int size )
{
    if( size < 1 || count < 1 )
    {
        return 0;
    }
    int idx = ( latestIndex + 1 - count ) % size;
    if( idx < 0 )
    {
        idx += size;
    }
    return idx;
}

inline Eigen::Matrix<ogTracker::realT, -1, 1> ogTracker::rmsPerMode( const Eigen::Matrix<realT, -1, -1> &projection )
{
    Eigen::Matrix<realT, -1, 1> rmsVals( projection.cols() );
    for( int mode = 0; mode < projection.cols(); ++mode )
    {
        rmsVals[mode] = std::sqrt( projection.col( mode ).array().square().mean() );
    }
    return rmsVals;
}

inline Eigen::Matrix<ogTracker::realT, -1, 1>
ogTracker::normalizeByReference( const Eigen::Matrix<realT, -1, 1> &rmsVals,
                                 const Eigen::Matrix<realT, -1, 1> &refVals,
                                 realT                               eps )
{
    const int nm = std::min( rmsVals.size(), refVals.size() );
    Eigen::Matrix<realT, -1, 1> normVals( nm );
    for( int mode = 0; mode < nm; ++mode )
    {
        if( std::abs( refVals[mode] ) > eps )
        {
            normVals[mode] = rmsVals[mode] / refVals[mode];
        }
        else
        {
            normVals[mode] = std::numeric_limits<realT>::quiet_NaN();
        }
    }
    return normVals;
}

inline void ogTracker::setupConfig()
{
    config.add( "calib.root", // TODO: this is really a MagAO-X configuration thing
                "",
                "calib.root",
                argType::Required,
                "calib",
                "root",
                false,
                "string",
                "Sparkle calibration root directory." );

    config.add( "pca.bufferN",
                "",
                "pca.bufferN",
                argType::Required,
                "pca",
                "bufferN",
                false,
                "int",
                "Rolling frame buffer size for PCA metrics." );

    config.add( "pca.minSamples",
                "",
                "pca.minSamples",
                argType::Required,
                "pca",
                "minSamples",
                false,
                "int",
                "Minimum buffered frames before metric updates." );

    config.add( "pca.klipMax",
                "",
                "pca.klipMax",
                argType::Required,
                "pca",
                "klipMax",
                false,
                "int",
                "Maximum number of PCA modes to use." );
    config.add( "pca.ogAvgN",
                "",
                "pca.ogAvgN",
                argType::Required,
                "pca",
                "ogAvgN",
                false,
                "int",
                "Number of measurements to average for pca_og output." );

    TELEMETER_SETUP_CONFIG( config );
    SHMIMMONITOR_SETUP_CONFIG( config );
}

inline int ogTracker::loadConfigImpl( mx::app::appConfigurator &_config )
{
    _config( m_calibRoot, "calib.root" );
    _config( m_bufferN, "pca.bufferN" );
    _config( m_minSamples, "pca.minSamples" );
    _config( m_klipMax, "pca.klipMax" );
    _config( m_ogAvgN, "pca.ogAvgN" );

    if( m_bufferN < 1 )
    {
        m_bufferN = 1;
    }
    if( m_klipMax < 1 )
    {
        m_klipMax = 1;
    }
    if( m_minSamples < 1 )
    {
        m_minSamples = 1;
    }
    if( m_ogAvgN < 1 )
    {
        m_ogAvgN = 1;
    }
    m_minSamples = std::max( m_minSamples, 10 * m_klipMax );

    TELEMETER_LOAD_CONFIG( _config );
    m_maxInterval = 1.0; // Publish telemetry at 1 Hz.
    SHMIMMONITOR_LOAD_CONFIG( _config );

    m_modeEls.clear();
    m_modeEls.reserve( static_cast<size_t>( m_klipMax ) );
    for( int n = 0; n < m_klipMax; ++n )
    {
        m_modeEls.push_back( "mode" + std::to_string( n ) );
    }

    return 0;
}

inline void ogTracker::loadConfig()
{
    loadConfigImpl( config );
}

inline int ogTracker::appStartup()
{
    TELEMETER_APP_STARTUP;
    SHMIMMONITOR_APP_STARTUP;

    REG_INDI_SETPROP( m_indiP_sep, m_tweeterDevice, "separation" );
    REG_INDI_SETPROP( m_indiP_ang, m_tweeterDevice, "angle" );
    REG_INDI_SETPROP( m_indiP_amp, m_tweeterDevice, "amp" );
    REG_INDI_SETPROP( m_indiP_freq, m_tweeterDevice, "frequency" );
    REG_INDI_SETPROP( m_indiP_modulating, m_tweeterDevice, "modulating" );

    createROIndiText( m_indiP_calibFolder, "calib_folder", "name", "Calibration Folder", "PCA", "Exact match folder" );
    createROIndiText( m_indiP_calibError, "calib_error", "state", "Calibration Error", "PCA", "Calibration status" );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_calibLoaded, "calib_loaded", "Calibration Loaded", "PCA" );
    m_indiP_calibLoaded.add( pcf::IndiElement( "current", 0 ) );

    CREATE_REG_INDI_RO_NUMBER( m_indiP_buffer, "buffer", "Rolling Buffer", "PCA" );
    m_indiP_buffer.add( pcf::IndiElement( "count", 0 ) );
    m_indiP_buffer.add( pcf::IndiElement( "capacity", 0 ) );

    createStandardIndiNumber<int>( m_indiP_ogAvgN, "ogAvgN", 1, 1000000, 1, "%d", "pca_og average count", "PCA" );
    m_indiP_ogAvgN["current"] = m_ogAvgN;
    m_indiP_ogAvgN["target"]  = m_ogAvgN;
    if( registerIndiPropertyNew( m_indiP_ogAvgN, INDI_NEWCALLBACK( m_indiP_ogAvgN ) ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    //CREATE_REG_INDI_RO_NUMBER( m_indiP_pcaRms, "pca_rms", "Rolling PCA RMS", "PCA" );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_pcaOG, "pca_og", "Instantaneous PCA RMS / Ref RMS", "PCA" );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_pcaOGAvg, "pca_og_avg", "Running-average PCA RMS / Ref RMS", "PCA" );
    CREATE_REG_INDI_RO_NUMBER( m_indiP_pcaOGSummary, "og_summary", "Mean pca_og_avg across modes <= 1", "PCA" );
    for( const auto &el : m_modeEls )
    {
        //m_indiP_pcaRms.add( pcf::IndiElement( el, 0 ) );
        m_indiP_pcaOG.add( pcf::IndiElement( el, 0 ) );
        m_indiP_pcaOGAvg.add( pcf::IndiElement( el, 0 ) );
    }
    m_indiP_pcaOGSummary.add( pcf::IndiElement( "current", 0 ) );

    m_indiP_calibFolder["name"] = "";
    m_indiP_calibError["state"] = "uninitialized";

    m_computeRun    = true;
    m_computeThread = std::thread( &ogTracker::computeThreadExec, this );

    state( stateCodes::OPERATING );
    return 0;
}

inline int ogTracker::setupFrameCircBuffLocked( int pixels )
{
    if( pixels <= 0 )
    {
        return -1;
    }

    if( shmimMonitorT::m_depth < 1 )
    {
        return -1;
    }

    // setting up a circBuff struct
    m_frameCircBuff = frameCircBuffT();

    // setting the max entries to the depth of the shmim
    const int depthCap = static_cast<int>( shmimMonitorT::m_depth );
    m_bufferCapacity   = std::max( 1, std::min( m_bufferN, depthCap ) );
    m_frameCircBuff.maxEntries( static_cast<cbIndexT>( m_bufferCapacity ) );

    m_metricsValid   = false;
    m_computePending = true;
    return 0;
}

inline int ogTracker::setupOgAverageLocked( int modes )
{
    if( modes <= 0 || m_ogAvgN <= 0 )
    {
        return -1;
    }

    m_ogAvgHistory.resize( m_ogAvgN, modes );
    m_ogAvgHistory.setZero();
    m_ogAvgSum.resize( modes );
    m_ogAvgSum.setZero();
    m_latestOgAvg.resize( modes );
    m_latestOgAvg.setZero();
    m_latestOgSummary = 0;
    m_ogAvgWrite = 0;
    m_ogAvgCount = 0;
    return 0;
}

inline void ogTracker::setCalibErrorLocked( const std::string &msg, logPrioT prio )
{
    if( m_calibError != msg )
    {
        m_calibError = msg;
        log<text_log>( "ogTracker calibration state: " + msg, prio );
    }
}

inline int ogTracker::loadCalibrationFiles( const std::filesystem::path &folderPath )
{
    const auto pcaPath = folderPath / "ref_pca.fits";
    const auto rmsPath = folderPath / "ref_rms.fits";

    if( !std::filesystem::exists( pcaPath ) )
    {
        setCalibErrorLocked( "missing ref_pca.fits", logPrio::LOG_WARNING );
        m_calibLoaded = false;
        return -1;
    }
    if( !std::filesystem::exists( rmsPath ) )
    {
        setCalibErrorLocked( "missing ref_rms.fits", logPrio::LOG_WARNING );
        m_calibLoaded = false;
        return -1;
    }

    mx::improc::eigenImage<realT> pcaRaw;
    mx::improc::eigenImage<realT> rmsRaw;
    mx::fits::fitsFile<realT>     ff;

    auto errc = ff.read( pcaRaw, pcaPath.string() );
    if( errc != mx::error_t::noerror )
    {
        setCalibErrorLocked( "failed reading ref_pca.fits", logPrio::LOG_ERROR );
        m_calibLoaded = false;
        return -1;
    }

    errc = ff.read( rmsRaw, rmsPath.string() );
    if( errc != mx::error_t::noerror )
    {
        setCalibErrorLocked( "failed reading ref_rms.fits", logPrio::LOG_ERROR );
        m_calibLoaded = false;
        return -1;
    }

    Eigen::Matrix<realT, -1, -1> pcaCanonical;
    if( pcaRaw.rows() >= pcaRaw.cols() )
    {
        pcaCanonical = pcaRaw;
    }
    else
    {
        pcaCanonical = pcaRaw.transpose();
    }

    if( pcaCanonical.cols() < 1 )
    {
        setCalibErrorLocked( "ref_pca.fits has zero modes", logPrio::LOG_ERROR );
        m_calibLoaded = false;
        return -1;
    }

    const int rmsCount = static_cast<int>( rmsRaw.size() );
    if( rmsCount < 1 )
    {
        setCalibErrorLocked( "ref_rms.fits has zero length", logPrio::LOG_ERROR );
        m_calibLoaded = false;
        return -1;
    }

    const int maxModesFromFiles = std::min( static_cast<int>( pcaCanonical.cols() ), rmsCount );
    m_activeModes               = std::min( m_klipMax, maxModesFromFiles );
    if( m_activeModes < 1 )
    {
        setCalibErrorLocked( "no overlapping modes", logPrio::LOG_ERROR );
        m_calibLoaded = false;
        return -1;
    }

    m_refPca = pcaCanonical.leftCols( m_activeModes );

    m_refRms.resize( m_activeModes );
    const realT *rmsPtr = rmsRaw.data();
    for( int n = 0; n < m_activeModes; ++n )
    {
        m_refRms[n] = rmsPtr[n];
    }

    if( m_framePixels > 0 && m_refPca.rows() != m_framePixels )
    {
        setCalibErrorLocked( "pixel mismatch with stream", logPrio::LOG_ERROR );
        m_calibLoaded = false;
        return -1;
    }

    m_latestRms.resize( m_activeModes );
    m_latestNorm.resize( m_activeModes );
    m_latestRms.setZero();
    m_latestNorm.setZero();
    setupOgAverageLocked( m_activeModes );
    m_metricsValid = false;

    setCalibErrorLocked( "ok", logPrio::LOG_NOTICE );
    m_calibLoaded = true;
    m_waitForParamChange = false;
    m_computePending = true;
    return 0;
}

inline int ogTracker::refreshCalibration()
{
    std::lock_guard<std::mutex> lock( m_dataMutex );
    m_paramsDirty = false;

    if( !m_modulating )
    {
        m_calibLoaded = false;
        setCalibErrorLocked( "sparkle not modulating", logPrio::LOG_INFO );
        return 0;
    }

    m_calibFolder = formatCalibFolder( m_sep, m_ang, m_amp, m_freq );

    const std::filesystem::path folderPath = std::filesystem::path( m_calibRoot ) / m_calibFolder;
    m_calibPath                         = folderPath.string();
    log<text_log>( "ogTracker looking for calibration folder: " + folderPath.string(), logPrio::LOG_NOTICE );
    if( !std::filesystem::exists( folderPath ) )
    {
        m_calibLoaded = false;
        setCalibErrorLocked( "missing calibration folder", logPrio::LOG_WARNING );
        m_waitForParamChange = true;
        return -1;
    }

    int rv = loadCalibrationFiles( folderPath );
    if( rv == 0 )
    {
        m_computePending = true;
    }
    return rv;
}

inline void ogTracker::computeMetricsFromSnapshot()
{
    Eigen::Matrix<realT, -1, -1, Eigen::RowMajor> frames;
    Eigen::Matrix<realT, -1, -1>                   refPca;
    Eigen::Matrix<realT, -1, 1>                    refRms;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_dataMutex );
        const int cbCount = static_cast<int>( m_frameCircBuff.size() );
        if( !m_modulating || m_waitForParamChange || !m_calibLoaded || cbCount < m_minSamples || m_activeModes < 1 )
        {
            return;
        }

        if( m_refPca.rows() != m_framePixels )
        {
            m_calibLoaded = false;
            setCalibErrorLocked( "pixel mismatch with stream", logPrio::LOG_ERROR );
            return;
        }

        const int latest = static_cast<int>( m_frameCircBuff.latest() );
        const int start  = cbWindowStartIndex( latest, cbCount, cbCount );
        frames.resize( cbCount, m_framePixels );
        for( int n = 0; n < cbCount; ++n )
        {
            realT *srcFrame =
                m_frameCircBuff.at( static_cast<cbIndexT>( start ), static_cast<cbIndexT>( n ) );
            if( srcFrame == nullptr )
            {
                return;
            }
            Eigen::Map<Eigen::Matrix<realT, -1, 1>> flat( srcFrame, m_framePixels );
            frames.row( n ) = flat.transpose();
        }

        refPca = m_refPca;
        refRms = m_refRms;
    }

    Eigen::Matrix<realT, 1, -1> meanFrame = frames.colwise().mean();
    frames.rowwise() -= meanFrame;

    const Eigen::Matrix<realT, -1, -1> proj = frames * refPca; // [ringCount, modes]
    const auto                          rms  = rmsPerMode( proj );
    const auto                          norm = normalizeByReference( rms, refRms, static_cast<realT>( 1e-8 ) );

    std::lock_guard<std::mutex> lock( m_dataMutex );
    m_latestRms   = rms;
    m_latestNorm  = norm;
    if( m_activeModes > 0 && m_ogAvgN > 0 )
    {
        if( m_ogAvgHistory.rows() != m_ogAvgN || m_ogAvgHistory.cols() != m_activeModes )
        {
            setupOgAverageLocked( m_activeModes );
        }

        if( m_ogAvgCount < m_ogAvgN )
        {
            m_ogAvgHistory.row( m_ogAvgWrite ) = norm.transpose();
            m_ogAvgSum += norm;
            ++m_ogAvgCount;
            m_ogAvgWrite = ( m_ogAvgWrite + 1 ) % m_ogAvgN;
        }
        else
        {
            m_ogAvgSum -= m_ogAvgHistory.row( m_ogAvgWrite ).transpose();
            m_ogAvgHistory.row( m_ogAvgWrite ) = norm.transpose();
            m_ogAvgSum += norm;
            m_ogAvgWrite = ( m_ogAvgWrite + 1 ) % m_ogAvgN;
        }

        if( m_ogAvgCount > 0 )
        {
            m_latestOgAvg = m_ogAvgSum / static_cast<realT>( m_ogAvgCount );
        }

        realT sumSummary = 0;
        int   nSummary   = 0;
        for( int n = 0; n < m_activeModes; ++n )
        {
            const realT v = m_latestOgAvg[n];
            if( std::isfinite( v ) && v <= static_cast<realT>( 1.0 ) )
            {
                sumSummary += v;
                ++nSummary;
            }
        }
        if( nSummary > 0 )
        {
            m_latestOgSummary = sumSummary / static_cast<realT>( nSummary );
        }
        else
        {
            m_latestOgSummary = 0;
        }
    }
    m_metricsValid = true;
}

inline void ogTracker::computeThreadExec()
{
    while( true )
    {
        { //mutex scope
            std::unique_lock<std::mutex> lock( m_dataMutex );
            m_computeCv.wait( lock, [this]() { return !m_computeRun || m_computePending; } );
            if( !m_computeRun )
            {
                break;
            }
            m_computePending = false;
        }

        computeMetricsFromSnapshot();
    }
}

inline int ogTracker::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> lock( m_dataMutex );
    m_frameWidth  = static_cast<int>( shmimMonitorT::m_width );
    m_frameHeight = static_cast<int>( shmimMonitorT::m_height );
    m_framePixels = m_frameWidth * m_frameHeight;

    if( shmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, "unsupported data type: expected float stream" } );
    }
    if( setupFrameCircBuffLocked( m_framePixels ) < 0 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "invalid stream geometry/depth for pointer circular buffer" } );
    }

    std::cerr << "connected to " << shmimMonitorT::m_shmimName << " " << shmimMonitorT::m_width << " "
              << shmimMonitorT::m_height << " " << shmimMonitorT::m_depth << "\n";

    if( m_refPca.rows() > 0 && m_refPca.rows() != m_framePixels )
    {
        m_calibLoaded = false;
        setCalibErrorLocked( "pixel mismatch with stream", logPrio::LOG_ERROR );
    }

    return 0;
}

inline int ogTracker::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    std::lock_guard<std::mutex> lock( m_dataMutex );
    if( m_framePixels <= 0 || !m_modulating || m_waitForParamChange || !m_calibLoaded )
    {
        return 0;
    }

    float *f_src = reinterpret_cast<realT *>( curr_src );
    m_frameCircBuff.nextEntry( f_src );
    m_computePending = true;
    m_computeCv.notify_one();
    return 0;
}

inline int ogTracker::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    SHMIMMONITOR_UPDATE_INDI;

    if( shmimMonitorT::m_smState == dev::shmimMonitorState::notfound )
    {
        if( !m_streamMissingLogged )
        {
            log<text_log>( "ogTracker stream not found: " + shmimMonitorT::m_shmimName +
                               " (polling until available)",
                           logPrio::LOG_NOTICE );
            m_streamMissingLogged = true;
        }
    }
    else if( shmimMonitorT::m_smState == dev::shmimMonitorState::connected && m_streamMissingLogged )
    {
        log<text_log>( "ogTracker stream connected: " + shmimMonitorT::m_shmimName, logPrio::LOG_NOTICE );
        m_streamMissingLogged = false;
    }

    bool paramsDirty = false;
    bool modulating  = false;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_dataMutex );
        paramsDirty = m_paramsDirty;
        modulating  = m_modulating;
    }

    if( !modulating )
    {
        if( !m_waitModulationLogged )
        {
            log<text_log>( "ogTracker waiting for tweeterSpeck modulation to turn ON; processing is paused.",
                           logPrio::LOG_NOTICE );
            m_waitModulationLogged = true;
        }
        std::lock_guard<std::mutex> lock( m_dataMutex );
        m_calibLoaded = false;
        setCalibErrorLocked( "sparkle not modulating", logPrio::LOG_INFO );
    }
    else if( paramsDirty )
    {
        if( m_waitModulationLogged )
        {
            log<text_log>( "ogTracker detected modulation ON; resuming calibration/processing.", logPrio::LOG_NOTICE );
            m_waitModulationLogged = false;
        }
        refreshCalibration();
    }

    recordDmSpeck( false );
    recordOgModes( false );
    TELEMETER_APP_LOGIC;

    std::lock_guard<std::mutex> lock( m_dataMutex );

    updateIfChanged( m_indiP_calibFolder, "name", m_calibFolder );
    updateIfChanged( m_indiP_calibError, "state", m_calibError );
    updateIfChanged( m_indiP_calibLoaded, "current", m_calibLoaded ? 1.0 : 0.0 );
    updatesIfChanged<double>(
        m_indiP_buffer,
        { "count", "capacity" },
        { static_cast<double>( m_frameCircBuff.size() ), static_cast<double>( m_bufferCapacity ) } );

    std::vector<double> normInstantOut( static_cast<size_t>( m_klipMax ), 0.0 );
    std::vector<double> normAvgOut( static_cast<size_t>( m_klipMax ), 0.0 );
    std::vector<const char *> modeElNames;
    modeElNames.reserve( m_modeEls.size() );
    for( const auto &el : m_modeEls )
    {
        modeElNames.push_back( el.c_str() );
    }
    if( m_metricsValid )
    {
        for( int n = 0; n < m_activeModes; ++n )
        {
            normInstantOut[static_cast<size_t>( n )] = static_cast<double>( m_latestNorm[n] );
            normAvgOut[static_cast<size_t>( n )]     = static_cast<double>( m_latestOgAvg[n] );
        }
    }

    //updatesIfChanged<double>( m_indiP_pcaRms, modeElNames, rmsOut );
    updatesIfChanged<double>( m_indiP_pcaOG, modeElNames, normInstantOut );
    updatesIfChanged<double>( m_indiP_pcaOGAvg, modeElNames, normAvgOut );
    updateIfChanged( m_indiP_pcaOGSummary, "current", static_cast<double>( m_latestOgSummary ) );
    updateIfChanged( m_indiP_ogAvgN, "current", m_ogAvgN, INDI_IDLE );
    updateIfChanged( m_indiP_ogAvgN, "target", m_ogAvgN, INDI_IDLE );

    return 0;
}

inline int ogTracker::appShutdown()
{
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_dataMutex );
        m_computeRun = false;
    }
    m_computeCv.notify_all();
    if( m_computeThread.joinable() )
    {
        m_computeThread.join();
    }

    TELEMETER_APP_SHUTDOWN;
    SHMIMMONITOR_APP_SHUTDOWN;
    return 0;
}

inline int ogTracker::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_dmspeck(), telem_dmmodes() );
}

inline int ogTracker::recordTelem( const telem_dmspeck * )
{
    return recordDmSpeck( true );
}

inline int ogTracker::recordTelem( const telem_dmmodes * )
{
    return recordOgModes( true );
}

inline int ogTracker::recordDmSpeck( bool force )
{
    bool        modulating;
    float       sep;
    float       ang;
    float       amp;
    float       freq;
    std::string calibFolder;
    std::string calibPath;
    std::string calibError;
    bool        calibLoaded;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_dataMutex );
        modulating = m_modulating;
        sep        = m_sep;
        ang        = m_ang;
        amp        = m_amp;
        freq       = m_freq;
        calibFolder = m_calibFolder;
        calibPath   = m_calibPath;
        calibError  = m_calibError;
        calibLoaded = m_calibLoaded;
    }

    static bool        lastModulating = false;
    static float       lastSep        = std::numeric_limits<float>::quiet_NaN();
    static float       lastAng        = std::numeric_limits<float>::quiet_NaN();
    static float       lastAmp        = std::numeric_limits<float>::quiet_NaN();
    static float       lastFreq       = std::numeric_limits<float>::quiet_NaN();
    static std::string lastCalibFolder;
    static std::string lastCalibPath;
    static std::string lastCalibError;
    static bool        lastCalibLoaded = false;
    constexpr float    floatEps        = 1e-6f;

    auto floatChanged = []( float oldV, float newV, float eps ) {
        if( std::isnan( oldV ) && std::isnan( newV ) )
        {
            return false;
        }
        if( std::isnan( oldV ) || std::isnan( newV ) )
        {
            return true;
        }
        return std::abs( oldV - newV ) > eps;
    };

    const bool changed = ( lastModulating != modulating ) || floatChanged( lastSep, sep, floatEps ) ||
                         floatChanged( lastAng, ang, floatEps ) || floatChanged( lastAmp, amp, floatEps ) ||
                         floatChanged( lastFreq, freq, floatEps ) || ( lastCalibFolder != calibFolder ) ||
                         ( lastCalibPath != calibPath ) || ( lastCalibError != calibError ) ||
                         ( lastCalibLoaded != calibLoaded );

    if( changed || force )
    {
        telem<telem_dmspeck>(
            { modulating, false, freq, { sep }, { ang }, { amp }, std::vector<bool>( { false } ) } );

        if( ( lastCalibPath != calibPath ) || ( lastCalibError != calibError ) || ( lastCalibLoaded != calibLoaded ) )
        {
            const std::string pathMsg = calibPath.empty() ? "no calibration folder resolved" : calibPath;
            log<text_log>( "ogTracker calibration path/status: " + pathMsg + " | state=" + calibError +
                               " | loaded=" + ( calibLoaded ? "1" : "0" ),
                           logPrio::LOG_NOTICE );
        }

        lastModulating = modulating;
        lastSep        = sep;
        lastAng        = ang;
        lastAmp        = amp;
        lastFreq       = freq;
        lastCalibFolder = calibFolder;
        lastCalibPath   = calibPath;
        lastCalibError  = calibError;
        lastCalibLoaded = calibLoaded;
    }

    return 0;
}

inline int ogTracker::recordOgModes( bool force )
{
    bool               metricsValid;
    std::vector<float> ogModeAvg;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_dataMutex );
        metricsValid        = m_metricsValid;
        const int modeCount = std::max( 0, m_activeModes );
        if( metricsValid && m_latestOgAvg.size() >= modeCount )
        {
            ogModeAvg.resize( static_cast<size_t>( modeCount ), 0 );
            for( int n = 0; n < modeCount; ++n )
            {
                ogModeAvg[static_cast<size_t>( n )] = m_latestOgAvg[n];
            }
        }
    }

    float summary = std::numeric_limits<float>::quiet_NaN();
    if( !ogModeAvg.empty() )
    {
        float sum = 0;
        int   nOk = 0;
        for( float v : ogModeAvg )
        {
            if( std::isfinite( v ) )
            {
                sum += v;
                ++nOk;
            }
        }
        if( nOk > 0 )
        {
            summary = sum / static_cast<float>( nOk );
        }
    }

    // Publish convention: [0] = og summary, [1..N] = per-mode rolling averages.
    std::vector<float> publishVals;
    publishVals.reserve( ogModeAvg.size() + 1 );
    publishVals.push_back( summary );
    publishVals.insert( publishVals.end(), ogModeAvg.begin(), ogModeAvg.end() );

    static bool               lastMetricsValid = false;
    static std::vector<float> lastPublishVals;
    constexpr float           floatEps = 1e-6f;

    bool modeAvgChanged = ( lastPublishVals.size() != publishVals.size() );
    if( !modeAvgChanged )
    {
        for( size_t n = 0; n < publishVals.size(); ++n )
        {
            if( std::isnan( lastPublishVals[n] ) && std::isnan( publishVals[n] ) )
            {
                continue;
            }
            if( std::isnan( lastPublishVals[n] ) || std::isnan( publishVals[n] ) )
            {
                modeAvgChanged = true;
                break;
            }
            if( std::abs( lastPublishVals[n] - publishVals[n] ) <= floatEps )
            {
                continue;
            }

            modeAvgChanged = true;
            break;
        }
    }

    if( modeAvgChanged || ( lastMetricsValid != metricsValid ) || force )
    {
        telem<telem_dmmodes>( publishVals );
        lastPublishVals  = publishVals;
        lastMetricsValid = metricsValid;
    }

    return 0;
}

INDI_SETCALLBACK_DEFN( ogTracker, m_indiP_sep )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sep, ipRecv );
    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_dataMutex );
        const float nextSep = ipRecv["current"].get<float>();
        if( std::abs( nextSep - m_sep ) > 1e-6f )
        {
            m_sep               = nextSep;
            m_paramsDirty       = true;
            m_waitForParamChange = false;
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( ogTracker, m_indiP_ang )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_ang, ipRecv );
    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_dataMutex );
        const float nextAng = ipRecv["current"].get<float>();
        if( std::abs( nextAng - m_ang ) > 1e-6f )
        {
            m_ang               = nextAng;
            m_paramsDirty       = true;
            m_waitForParamChange = false;
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( ogTracker, m_indiP_amp )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_amp, ipRecv );
    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_dataMutex );
        const float nextAmp = ipRecv["current"].get<float>();
        if( std::abs( nextAmp - m_amp ) > 1e-6f )
        {
            m_amp               = nextAmp;
            m_paramsDirty       = true;
            m_waitForParamChange = false;
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( ogTracker, m_indiP_freq )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_freq, ipRecv );
    if( ipRecv.find( "current" ) )
    {
        std::lock_guard<std::mutex> lock( m_dataMutex );
        const float nextFreq = ipRecv["current"].get<float>();
        if( std::abs( nextFreq - m_freq ) > 1e-6f )
        {
            m_freq              = nextFreq;
            m_paramsDirty       = true;
            m_waitForParamChange = false;
        }
    }
    return 0;
}

INDI_SETCALLBACK_DEFN( ogTracker, m_indiP_modulating )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_modulating, ipRecv );
    if( ipRecv.find( "toggle" ) )
    {
        std::lock_guard<std::mutex> lock( m_dataMutex );
        const bool nextModulating = ( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On );
        if( nextModulating != m_modulating )
        {
            m_modulating  = nextModulating;
            m_paramsDirty = true;
            if( m_modulating )
            {
                m_waitForParamChange = false;
            }
        }
    }
    return 0;
}

INDI_NEWCALLBACK_DEFN( ogTracker, m_indiP_ogAvgN )( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getName() != m_indiP_ogAvgN.getName() )
    {
        log<software_error>( { __FILE__, __LINE__, "invalid indi property received" } );
        return -1;
    }

    int target;
    if( indiTargetUpdate( m_indiP_ogAvgN, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    std::lock_guard<std::mutex> lock( m_dataMutex );
    if( target < 1 )
    {
        target = 1;
    }
    if( target != m_ogAvgN )
    {
        m_ogAvgN = target;
        if( m_activeModes > 0 )
        {
            setupOgAverageLocked( m_activeModes );
        }
        log<text_log>( "set pca_og averaging count to " + std::to_string( m_ogAvgN ), logPrio::LOG_NOTICE );
    }

    updateIfChanged( m_indiP_ogAvgN, "current", m_ogAvgN, INDI_IDLE );
    updateIfChanged( m_indiP_ogAvgN, "target", m_ogAvgN, INDI_IDLE );
    return 0;
}

} // namespace app
} // namespace MagAOX

#endif // ogTracker_hpp
