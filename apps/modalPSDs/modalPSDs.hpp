/** \file modalPSDs.hpp
 * \brief The MagAO-X modalPSDs app header file
 *
 * \ingroup modalPSDs_files
 */

#ifndef modalPSDs_hpp
#define modalPSDs_hpp

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#include <atomic>
#include <condition_variable>
#include <mx/sigproc/circularBuffer.hpp>
#include <mx/sigproc/signalWindows.hpp>

#include <mx/math/ft/fftwEnvironment.hpp>
#include <mx/math/ft/fftT.hpp>

/** \defgroup modalPSDs
 * \brief An application to calculate rolling PSDs of modal amplitudes
 *
 * <a href="../handbook/operating/software/apps/modalPSDs.html">Application Documentation</a>
 *
 * \ingroup apps
 *
 */

/** \defgroup modalPSDs_files
 * \ingroup modalPSDs
 */

namespace MagAOX
{
namespace app
{

/// Class for application to calculate rolling PSDs of modal amplitudes.
/**
 * \ingroup modalPSDs
 */
class modalPSDs : public MagAOXApp<true>, public dev::shmimMonitor<modalPSDs>
{
    // Give the test harness access.
    friend class modalPSDs_test;

    friend class dev::shmimMonitor<modalPSDs>;

  public:
    typedef float realT;

    typedef std::complex<realT> complexT;

    typedef int32_t cbIndexT; ///< The index for the circular buffer

    /// The base shmimMonitor type
    typedef dev::shmimMonitor<modalPSDs> shmimMonitorT;

    /// The amplitude circular buffer type
    typedef mx::sigproc::circularBufferIndex<realT *, cbIndexT, true> ampCircBuffT;

  protected:
    /** \name Configurable Parameters
     *@{
     */

    int m_loopNumber{ 1 };

    std::string m_shmimBase; /**< The base name for the PSD shmims.  If empty, the default,
                                  then aolN where N is the loop number is used*/

    std::string m_shmimTag{ "cl" }; /**< The tag to apply to the front of psds in the shmim name.  Default is cl.  */

    std::string m_fpsDevice;               ///< Device name for getting fps to set circular buffer length.
    std::string m_fpsProperty{ "fps" };    ///< Property name for getting fps to set circular buffer length.
    std::string m_fpsElement{ "current" }; ///< Element name for getting fps to set circular buffer length.

    std::string m_loopStateDevice;                   ///< Optional device name providing loop-state gating updates.
    std::string m_loopStateProperty{ "loop_state" }; ///< Optional property name providing loop-state gating updates.
    std::string m_loopStateElement{ "toggle" };      ///< Element name used to interpret closed-loop state.

    bool m_useLoopState{ false }; ///< Whether PSD ingestion is gated by an external loop-state property.

    std::atomic<bool> m_loopClosed{ true }; ///< Current closed-loop state used to gate PSD ingestion and processing.

    realT m_fpsTol{ 0 }; ///< The tolerance for detecting a change in FPS.

    std::atomic<realT> m_psdTime{ 1 };     ///< The length of time over which to calculate PSDs.  The default is 1 sec.
    std::atomic<realT> m_psdAvgTime{ 10 }; ///< The time over which to average PSD estimates.  The default is 10 sec.
    std::atomic<realT> m_meanTime{
        10 }; ///< The time over which to calculate the mean for detrending.  The default is 10 sec.

    // realT m_overSize {10}; ///< Multiplicative factor by which to oversize the circular buffer, to give good mean
    // estimates and account for time-to-calculate.

    realT m_psdOverlapFraction{ 0.5 }; ///< The fraction of the sample time to overlap by.

    int m_nPSDHistory{ 100 }; ///< Minimum number of raw PSD estimates to retain in the history stream.

    ///@}

    size_t m_nModes{ 0 }; ///< the number of modes to calculate PSDs for.

    ampCircBuffT m_ampCircBuff;

    // std::vector<ampCircBuffT> m_ampCircBuffs;

    std::atomic<realT> m_fps{ 0 }; ///< The current input frame rate used to size PSD windows.

    realT m_df{ 1 };

    cbIndexT m_tsSize{ 2000 }; ///< The length of the time series sample over which the PSD is calculated

    cbIndexT m_tsOverlapSize{ 1000 }; ///< The number of samples in the overlap

    cbIndexT m_meanSize{ 20000 }; ///< The length of the time series over which to calculate the mean

    std::vector<realT> m_win; ///< The window function.  By default this is Hann.

    std::vector<realT *> m_tsPtrs;   ///< Snapshot of the latest time-series window pointers for PSD computation.
    std::vector<realT *> m_meanPtrs; ///< Snapshot of the latest mean-window pointers for PSD computation.

    realT *m_tsWork{ nullptr };
    size_t m_tsWorkSize{ 0 };

    std::complex<realT> *m_fftWork{ nullptr };
    size_t               m_fftWorkSize{ 0 };

    std::vector<realT> m_psd;

    mx::math::ft::fftT<realT, std::complex<realT>, 1, 0> m_fft;
    mx::math::ft::fftwEnvironment<realT>                 m_fftEnv;

    /** \name PSD Calculation Thread
     * Handling of offloads from the average woofer shape
     * @{
     */
    int m_psdThreadPrio{ 0 }; ///< Priority of the PSD Calculation thread.

    std::string m_psdThreadCpuset; ///< The cpuset to use for the PSD Calculation thread.

    std::thread m_psdThread; ///< The PSD Calculation thread.

    bool m_psdThreadInit{ true }; ///< Initialization flag for the PSD Calculation thread.

    std::atomic<bool> m_psdRestarting{ true }; /**< Synchronization flag.  This will only become false
                                                    after a successful call to allocate.*/

    bool m_psdWaiting{ false }; ///< Synchronization flag protected by m_psdMutex.

    std::mutex m_psdMutex; ///< Protects restart handoff between allocate() and the PSD thread.

    std::condition_variable m_psdCond; ///< Coordinates quiescence during restart.

    pid_t m_psdThreadID{ 0 }; ///< PSD Calculation thread PID.

    pcf::IndiProperty m_psdThreadProp; ///< The property to hold the PSD Calculation thread details.

    /// PS Calculation thread starter function
    static void psdThreadStart( modalPSDs *p /**< [in] pointer to this */ );

    /// PSD Calculation thread function
    /** Runs until m_shutdown is true.
     */
    void psdThreadExec();

    /// Compute the reference entry for the latest logical window while preserving historical semantics.
    /** The returned window ends at the sample immediately preceding snapshot.latest. This matches the
     *  longstanding modalPSDs behavior and keeps the writable/latest publication edge out of the PSD window.
     */
    static cbIndexT latestWindowRefEntry( const ampCircBuffT::snapshotT &sn,   ///< [in] the snapshot to use
                                          cbIndexT                       count ///< [in] the requested window length
    );

    /// Compute the reference entry for a logical window immediately preceding another logical window.
    static cbIndexT precedingWindowRefEntry( const ampCircBuffT::snapshotT &sn, ///< [in] the snapshot to use
                                             cbIndexT refEntry,                 ///< [in] later window reference entry
                                             cbIndexT count                     ///< [in] the requested window length
    );

    /// Compute the forward logical advance between two circular-buffer reference entries.
    static cbIndexT circularEntryAdvance( cbIndexT from,      ///< [in] earlier logical entry
                                          cbIndexT to,        ///< [in] later logical entry
                                          cbIndexT maxEntries ///< [in] circular-buffer capacity
    );

    /// Load the PSD and mean pointer windows from a single validated snapshot.
    bool loadPsdInputWindows( ampCircBuffT::snapshotT &sn ///< [out] the snapshot used for both windows
    );

    /// Recompute the per-mode sums for the full mean window from the currently loaded pointers.
    void recomputeMeanSums( std::vector<double> &meanSums /**< [out] per-mode sums over the full mean window */ ) const;

    /// Update per-mode mean sums using the cached oldest slice and the newest loaded mean-window slice.
    void
    rollMeanSums( std::vector<double>      &meanSums,      /**< [in,out] per-mode sums to update */
                  const std::vector<realT> &meanHeadCache, /**< [in] cached oldest slice from the prior mean window */
                  cbIndexT                  advance /**< [in] number of samples by which the mean window advanced */
    ) const;

    /// Cache the oldest slice of the currently loaded mean window for the next rolling-mean update.
    void cacheMeanHead( std::vector<realT> &meanHeadCache, /**< [out] storage for the cached oldest slice */
                        cbIndexT            count          /**< [in] number of mean-window samples to cache */
    ) const;

    /// Count how many raw PSD planes are currently retained across the published and overflow histories.
    uint64_t storedRawPSDCount() const;

    /// Locate a retained raw PSD plane by age, where age 0 is the newest plane.
    const realT *rawPSDPlaneByAge( uint64_t age,          /**< [in] age of the requested raw PSD plane */
                                   size_t   planeElements /**< [in] number of elements per raw PSD plane */
    ) const;

    /// Recompute the averaged-PSD running sum from the newest `windowCount` retained raw PSD planes.
    void recomputeAveragedPSDSum( std::vector<double> &avgPsdSum,    /**< [out] running sum over the averaging window */
                                  uint64_t             windowCount,  /**< [in] number of raw PSD planes to sum */
                                  size_t               planeElements /**< [in] number of elements per raw PSD plane */
    ) const;

    /// Add one raw PSD plane and optionally subtract one outgoing raw PSD plane from the running average sum.
    static void updatePlaneSum( std::vector<double> &planeSum, /**< [in,out] running sum to update */
                                const realT         *addPlane, /**< [in] newest raw PSD plane to add */
                                const realT *removePlane,  /**< [in] outgoing raw PSD plane to subtract, or nullptr */
                                size_t       planeElements /**< [in] number of elements per raw PSD plane */
    );

    /// Determine whether incoming frames should currently be accepted into the PSD history.
    bool acceptLoopStateFrame() const;

    /// Calculate how many raw PSD estimates are needed to cover the requested averaging time.
    int desiredPSDAverageCount() const;

    /// Calculate how many samples are needed for the mean-subtraction window at the current FPS.
    cbIndexT desiredMeanSampleCount( realT fps /**< [in] frame rate used to convert mean time into samples */ ) const;

    /// Calculate the total input-history depth needed to read both windows safely from the fixed-size circular buffer.
    cbIndexT requiredInputHistoryDepth() const;

    /// Calculate the additional PSD history depth needed beyond the published raw-PSD shmim.
    uint32_t rawPSDHistoryDepth() const;

    /// Calculate the published raw PSD history depth retained in shared memory.
    uint32_t publishedRawPSDHistoryDepth() const;

    IMAGE *m_freqStream{ nullptr }; ///< The ImageStreamIO shared memory buffer to hold the frequency scale

    mx::improc::eigenImage<realT> m_psdBuffer;

    std::vector<realT> m_rawPSDHistory; ///< Heap-backed overflow history for raw PSD estimates beyond the shmim depth.

    uint32_t m_rawPSDHistoryDepth{ 0 }; ///< Number of overflow PSD estimates currently allocated in `m_rawPSDHistory`.

    uint32_t m_rawPSDHistoryNext{ 0 }; ///< Next overflow slot to overwrite in `m_rawPSDHistory`.

    uint64_t m_rawPSDHistoryCount{ 0 }; ///< Number of overflow PSD estimates stored since the last restart.

    IMAGE *m_rawpsdStream{ nullptr }; ///< The ImageStreamIO shared memory buffer to hold the raw psds

    IMAGE *m_avgpsdStream{ nullptr }; ///< The ImageStreamIO shared memory buffer to hold the average psds

  public:
    /// Default c'tor.
    modalPSDs();

    /// D'tor, declared and defined for noexcept.
    ~modalPSDs() noexcept
    {
    }

    virtual void setupConfig();

    /// Implementation of loadConfig logic, separated for testing.
    /** This is called by loadConfig().
     */
    int loadConfigImpl( mx::app::appConfigurator &_config /**< [in] an application configuration
                                                                    from which to load values*/ );

    virtual void loadConfig();

    /// Startup function
    /**
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for modalPSDs.
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

    // shmimMonitor Interface
  protected:
    int allocate( const dev::shmimT &dummy /**< [in] tag to differentiate shmimMonitor parents.*/ );

    int allocatePSDStreams();

    int processImage( void              *curr_src, ///< [in] pointer to start of current frame.
                      const dev::shmimT &dummy     ///< [in] tag to differentiate shmimMonitor parents.
    );

    // INDI Interface
  protected:
    pcf::IndiProperty m_indiP_psdTime;
    pcf::IndiProperty m_indiP_psdAvgTime;
    pcf::IndiProperty m_indiP_meanTime;
    pcf::IndiProperty m_indiP_overSize;
    pcf::IndiProperty m_indiP_fpsSource;
    pcf::IndiProperty m_indiP_loop;
    pcf::IndiProperty m_indiP_fps;

  public:
    INDI_NEWCALLBACK_DECL( modalPSDs, m_indiP_psdTime );
    INDI_NEWCALLBACK_DECL( modalPSDs, m_indiP_psdAvgTime );
    INDI_NEWCALLBACK_DECL( modalPSDs, m_indiP_meanTime );
    INDI_NEWCALLBACK_DECL( modalPSDs, m_indiP_overSize );
    INDI_SETCALLBACK_DECL( modalPSDs, m_indiP_fpsSource );
    INDI_SETCALLBACK_DECL( modalPSDs, m_indiP_loop );
};

modalPSDs::modalPSDs() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    return;
}

void modalPSDs::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );

    config.add( "loop.number",
                "",
                "loop.number",
                argType::Required,
                "loop",
                "number",
                false,
                "string",
                "Loop number, as in aolN" );

    config.add( "psds.shmimBase",
                "",
                "psds.shmimBase",
                argType::Required,
                "psds",
                "shmimBase",
                false,
                "string",
                "The base name for the PSD shmims.  If empty, the default, "
                "then aolN where N is the loop number is used" );

    config.add( "psds.shmimTag",
                "",
                "psds.shmimTag",
                argType::Required,
                "psds",
                "shmimTag",
                false,
                "string",
                "The tag to apply to the front of psds in the shmim name.  Default is cl. " );

    config.add( "circBuff.fpsDevice",
                "",
                "circBuff.fpsDevice",
                argType::Required,
                "circBuff",
                "fpsDevice",
                false,
                "string",
                "Device name for getting fps to set circular buffer length." );

    config.add( "circBuff.fpsProperty",
                "",
                "circBuff.fpsProperty",
                argType::Required,
                "circBuff",
                "fpsProperty",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'fps'." );

    config.add( "circBuff.fpsElement",
                "",
                "circBuff.fpsElement",
                argType::Required,
                "circBuff",
                "fpsElement",
                false,
                "string",
                "Property name for getting fps to set circular buffer length. Default is 'current'." );

    config.add( "circBuff.fpsTol",
                "",
                "circBuff.fpsTol",
                argType::Required,
                "circBuff",
                "fpsTol",
                false,
                "float",
                "Tolerance for detecting a change in FPS.  Default is 0." );

    config.add( "circBuff.defaultFPS",
                "",
                "circBuff.defaultFPS",
                argType::Required,
                "circBuff",
                "defaultFPS",
                false,
                "realT",
                "Default FPS at startup, will enable changing average length with psdTime before INDI available." );

    config.add( "circBuff.loopStateDevice",
                "",
                "circBuff.loopStateDevice",
                argType::Required,
                "circBuff",
                "loopStateDevice",
                false,
                "string",
                "Optional device name providing loop-state gating. If unset, PSDs ignore loop state." );

    config.add( "circBuff.loopStateProperty",
                "",
                "circBuff.loopStateProperty",
                argType::Required,
                "circBuff",
                "loopStateProperty",
                false,
                "string",
                "Optional property name providing loop-state gating. Default is 'loop_state'." );

    config.add( "circBuff.loopStateElement",
                "",
                "circBuff.loopStateElement",
                argType::Required,
                "circBuff",
                "loopStateElement",
                false,
                "string",
                "Element name interpreted as the closed-loop state. Default is 'toggle'." );

    config.add( "circBuff.psdTime",
                "",
                "circBuff.psdTime",
                argType::Required,
                "circBuff",
                "psdTime",
                false,
                "realT",
                "The length of time over which to calculate PSDs.  The default is 1 sec." );

    config.add( "circBuff.psdAvgTime",
                "",
                "circBuff.psdAvgTime",
                argType::Required,
                "circBuff",
                "psdAvgTime",
                false,
                "realT",
                "The length of time over which to average PSD estimates.  The default is 10 sec." );

    config.add( "circBuff.meanTime",
                "",
                "circBuff.meanTime",
                argType::Required,
                "circBuff",
                "meanTime",
                false,
                "realT",
                "The length of time over which to calculate the detrending mean.  The default is 10 sec." );
}

int modalPSDs::loadConfigImpl( mx::app::appConfigurator &_config )
{
    SHMIMMONITOR_LOAD_CONFIG( _config );

    _config( m_loopNumber, "loop.number" );

    m_shmimBase = "aol" + std::to_string( m_loopNumber );
    _config( m_shmimBase, "psds.shmimBase" );

    _config( m_shmimTag, "psds.shmimTag" );

    _config( m_fpsDevice, "circBuff.fpsDevice" );
    _config( m_fpsProperty, "circBuff.fpsProperty" );
    _config( m_fpsElement, "circBuff.fpsElement" );
    _config( m_fpsTol, "circBuff.fpsTol" );
    _config( m_loopStateDevice, "circBuff.loopStateDevice" );
    _config( m_loopStateProperty, "circBuff.loopStateProperty" );
    _config( m_loopStateElement, "circBuff.loopStateElement" );

    m_useLoopState = !m_loopStateDevice.empty();
    m_loopClosed.store( m_useLoopState == false, std::memory_order_release );

    realT psdTime = m_psdTime.load();
    _config( psdTime, "circBuff.psdTime" );
    m_psdTime.store( psdTime );

    realT psdAvgTime = m_psdAvgTime.load();
    _config( psdAvgTime, "circBuff.psdAvgTime" );
    m_psdAvgTime.store( psdAvgTime );

    realT meanTime = m_meanTime.load();
    _config( meanTime, "circBuff.meanTime" );
    m_meanTime.store( meanTime );

    return 0;
}

void modalPSDs::loadConfig()
{
    loadConfigImpl( config );
}

int modalPSDs::appStartup()
{
    CREATE_REG_INDI_NEW_NUMBERF( m_indiP_psdTime, "psdTime", 0, 60, 0.1, "%0.1f", "PSD time", "PSD Setup" );
    m_indiP_psdTime["current"].set( m_psdTime.load() );
    m_indiP_psdTime["target"].set( m_psdTime.load() );

    CREATE_REG_INDI_NEW_NUMBERU( m_indiP_psdAvgTime, "psdAvgTime", 0, 60, 0.1, "%0.1f", "PSD Avg. Time", "PSD Setup" );
    m_indiP_psdAvgTime["current"].set( m_psdAvgTime.load() );
    m_indiP_psdAvgTime["target"].set( m_psdAvgTime.load() );

    CREATE_REG_INDI_NEW_NUMBERU( m_indiP_meanTime, "meanTime", 0, 600, 0.1, "%0.1f", "Mean Time", "PSD Setup" );
    m_indiP_meanTime["current"].set( m_meanTime.load() );
    m_indiP_meanTime["target"].set( m_meanTime.load() );

    if( m_fpsDevice == "" )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "FPS source is not configurated (circBuff.fpsDevice)" } );
    }

    REG_INDI_SETPROP( m_indiP_fpsSource, m_fpsDevice, m_fpsProperty );

    if( m_useLoopState == true )
    {
        REG_INDI_SETPROP( m_indiP_loop, m_loopStateDevice, m_loopStateProperty );
    }

    CREATE_REG_INDI_RO_NUMBER( m_indiP_fps, "fps", "current", "Circular Buffer" );
    m_indiP_fps.add( pcf::IndiElement( "current" ) );
    m_indiP_fps["current"] = m_fps.load();

    SHMIMMONITOR_APP_STARTUP;

    XWCAPP_THREAD_START( m_psdThread,
                         m_psdThreadInit,
                         m_psdThreadID,
                         m_psdThreadProp,
                         m_psdThreadPrio,
                         m_psdThreadCpuset,
                         "psdcalc",
                         psdThreadStart );

    state( stateCodes::OPERATING );

    return 0;
}

int modalPSDs::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;

    XWCAPP_THREAD_CHECK( m_psdThread, "psdcalc" );

    std::unique_lock<std::mutex> lock( m_indiMutex );

    SHMIMMONITOR_UPDATE_INDI;

    return 0;
}

int modalPSDs::appShutdown()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_psdMutex );
        m_psdWaiting = false;
    }
    m_psdCond.notify_all();

    XWCAPP_THREAD_STOP( m_psdThread );

    SHMIMMONITOR_APP_SHUTDOWN;

    if( m_rawpsdStream )
    {
        ImageStreamIO_destroyIm( m_rawpsdStream );
        free( m_rawpsdStream );
    }

    if( m_avgpsdStream )
    {
        ImageStreamIO_destroyIm( m_avgpsdStream );
        free( m_avgpsdStream );
    }

    if( m_freqStream )
    {
        ImageStreamIO_destroyIm( m_freqStream );
        free( m_freqStream );
    }

    if( m_tsWork )
    {
        fftw_free( m_tsWork );
    }

    if( m_fftWork )
    {
        fftw_free( m_fftWork );
    }

    return 0;
}

int modalPSDs::allocate( const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    m_psdRestarting.store( true, std::memory_order_release );

    // Wait for FPS to become not 0
    // We wait indefinitely, the other process just might not be alive
    bool logged = false;
    while( m_fps.load( std::memory_order_acquire ) <= 0 && !shutdown() )
    {
        if( !logged ) // log every thirty seconds
        {
            log<text_log>( "waiting for FPS...", logPrio::LOG_NOTICE );
            logged = true;
        }
        mx::sys::sleep( 1 );
    }

    { // mutex scope
        std::unique_lock<std::mutex> lock( m_psdMutex );
        m_psdCond.wait( lock, [this]() { return m_psdWaiting || shutdown(); } );
    }

    std::cerr << "PSD waiting \n";

    if( shutdown() )
    {
        return 0; // If shutdown() is true then shmimMonitor will cleanup
    }

    // Check for unsupported type (must be realT)
    if( shmimMonitorT::m_dataType != IMAGESTRUCT_FLOAT )
    {
        // must be a vector of size 1 on one axis
        log<software_error>( { __FILE__, __LINE__, "unsupported data type: must be realT" } );
        return -1;
    }

    // Check for unexpected format
    if( shmimMonitorT::m_width != 1 && shmimMonitorT::m_height != 1 )
    {
        // must be a vector of size 1 on one axis
        log<software_error>( { __FILE__, __LINE__, "unexpected shmim format" } );
        return -1;
    }

    std::cerr << "connected to " << shmimMonitorT::m_shmimName << " " << shmimMonitorT::m_width << " "
              << shmimMonitorT::m_height << " " << shmimMonitorT::m_depth << "\n";

    m_nModes = shmimMonitorT::m_width * shmimMonitorT::m_height;

    realT fps     = m_fps.load( std::memory_order_acquire );
    realT psdTime = m_psdTime.load( std::memory_order_acquire );

    m_tsSize = fps * psdTime;

    // Adjust length if odd to ensure we get the Nyquist frequency
    if( m_tsSize % 2 == 1 )
    {
        m_tsSize += 1;
    }

    m_tsOverlapSize = m_tsSize * m_psdOverlapFraction;

    if( m_tsOverlapSize == 0 || !std::isnormal( m_tsOverlapSize ) )
    {
        log<software_error>(
            { __FILE__, __LINE__, "bad m_tsOverlapSize value: " + std::to_string( m_tsOverlapSize ) } );
        return -1;
    }

    m_meanSize = desiredMeanSampleCount( fps );

    if( static_cast<uint32_t>( m_tsSize + 2 ) >= shmimMonitorT::m_depth )
    {
        log<software_error>( { __FILE__, __LINE__, "input circ buff is not long enough for psd and mean windows" } );
        return -1;
    }

    cbIndexT maxMeanSize = shmimMonitorT::m_depth - m_tsSize - 2;
    if( m_meanSize > maxMeanSize )
    {
        log<text_log>( "input circ buff is not long enough for meanTime, truncating to " +
                           std::to_string( static_cast<double>( maxMeanSize ) / fps ) + " sec",
                       logPrio::LOG_WARNING );
        m_meanSize = maxMeanSize;
    }

    m_ampCircBuff.maxEntries( requiredInputHistoryDepth() );

    m_tsPtrs.resize( m_tsSize );
    m_meanPtrs.resize( m_meanSize );

    // Create the window
    m_win.resize( m_tsSize );
    mx::sigproc::window::hann( m_win );

    // Set up the FFT and working memory
    m_fft.plan( m_tsSize, mx::math::ft::dir::forward, false );

    if( m_tsWork )
    {
        fftw_free( m_tsWork );
    }
    m_tsWork = mx::math::ft::fftw_malloc<realT>( m_tsSize );

    if( m_fftWork )
    {
        fftw_free( m_fftWork );
    }

    m_fftWork = mx::math::ft::fftw_malloc<std::complex<realT>>( ( m_tsSize / 2 + 1 ) );

    m_psd.resize( m_tsSize / 2 + 1 );

    m_df = 1.0 / ( m_tsSize / fps );

    // Create the shared memory images
    uint32_t imsize[3];

    // First the frequency
    imsize[0] = 1;
    imsize[1] = m_psd.size();
    imsize[2] = 1;

    if( m_freqStream )
    {
        ImageStreamIO_destroyIm( m_freqStream );
        free( m_freqStream );
    }
    m_freqStream = static_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );

    ImageStreamIO_createIm_gpu( m_freqStream,
                                ( m_shmimBase + "_freq" ).c_str(),
                                3,
                                imsize,
                                IMAGESTRUCT_FLOAT,
                                -1,
                                1,
                                IMAGE_NB_SEMAPHORE,
                                0,
                                CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                0 );
    m_freqStream->md->write = 1;
    for( size_t n = 0; n < m_psd.size(); ++n )
    {
        m_freqStream->array.F[n] = n * m_df;
    }

    // Set the time of last write
    clock_gettime( CLOCK_REALTIME, &m_freqStream->md->writetime );
    m_freqStream->md->atime = m_freqStream->md->writetime;

    // Update cnt1
    m_freqStream->md->cnt1 = 0;

    // Update cnt0
    m_freqStream->md->cnt0 = 0;

    m_freqStream->md->write = 0;
    ImageStreamIO_sempost( m_freqStream, -1 );

    allocatePSDStreams();

    std::cerr << "done restarting\n";

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_psdMutex );
        m_psdWaiting = false;
        m_psdRestarting.store( false, std::memory_order_release );
    }
    m_psdCond.notify_all();

    return 0;
}

int modalPSDs::allocatePSDStreams()
{
    if( m_rawpsdStream )
    {
        ImageStreamIO_destroyIm( m_rawpsdStream );
        free( m_rawpsdStream );
    }

    uint32_t imsize[3];
    imsize[0] = m_psd.size();
    imsize[1] = m_nModes;
    imsize[2] = publishedRawPSDHistoryDepth();

    m_rawpsdStream = static_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );

    ImageStreamIO_createIm_gpu( m_rawpsdStream,
                                ( m_shmimBase + "_raw_" + m_shmimTag + "psds" ).c_str(),
                                3,
                                imsize,
                                IMAGESTRUCT_FLOAT,
                                -1,
                                1,
                                IMAGE_NB_SEMAPHORE,
                                0,
                                CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                0 );

    if( m_avgpsdStream )
    {
        ImageStreamIO_destroyIm( m_avgpsdStream );
        free( m_avgpsdStream );
    }

    imsize[0] = m_psd.size();
    imsize[1] = m_nModes;
    imsize[2] = 1;

    m_avgpsdStream = static_cast<IMAGE *>( malloc( sizeof( IMAGE ) ) );
    ImageStreamIO_createIm_gpu( m_avgpsdStream,
                                ( m_shmimBase + "_" + m_shmimTag + "psds" ).c_str(),
                                3,
                                imsize,
                                IMAGESTRUCT_FLOAT,
                                -1,
                                1,
                                IMAGE_NB_SEMAPHORE,
                                0,
                                CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                0 );

    m_psdBuffer.resize( m_psd.size(), m_nModes );

    size_t planeElements = m_psdBuffer.rows() * m_psdBuffer.cols();
    m_rawPSDHistoryDepth = rawPSDHistoryDepth();
    m_rawPSDHistory.assign( planeElements * static_cast<size_t>( m_rawPSDHistoryDepth ), 0 );
    m_rawPSDHistoryNext  = 0;
    m_rawPSDHistoryCount = 0;

    return 0;
}

int modalPSDs::processImage( void *curr_src, const dev::shmimT &dummy )
{
    static_cast<void>( dummy );

    if( acceptLoopStateFrame() == false )
    {
        return 0;
    }

    float *f_src = static_cast<float *>( curr_src );

    m_ampCircBuff.nextEntry( f_src );

    return 0;
}

void modalPSDs::psdThreadStart( modalPSDs *p )
{
    p->psdThreadExec();
}

void modalPSDs::psdThreadExec()
{
    m_psdThreadID = syscall( SYS_gettid );

    std::vector<double>     meanSums;
    std::vector<realT>      meanHeadCache;
    ampCircBuffT::snapshotT prevSnap;
    cbIndexT                prevMeanRefEntry = 0;
    bool                    haveMeanSums     = false;
    std::vector<double>     avgPsdSum;
    uint64_t                avgPsdWindowCount = 0;

    while( m_psdThreadInit == true && shutdown() == 0 )
    {
        sleep( 1 );
    }

    while( shutdown() == 0 )
    {
        { // mutex scope
            std::unique_lock<std::mutex> lock( m_psdMutex );

            if( m_psdRestarting.load( std::memory_order_acquire ) == true || m_ampCircBuff.maxEntries() == 0 )
            {
                m_psdWaiting = true;
                m_psdCond.notify_all();
            }

            m_psdCond.wait( lock,
                            [this]()
                            {
                                return shutdown() || ( m_psdRestarting.load( std::memory_order_acquire ) == false &&
                                                       m_ampCircBuff.maxEntries() != 0 );
                            } );

            m_psdWaiting = false;
        }

        if( shutdown() )
        {
            break;
        }

        if( m_ampCircBuff.maxEntries() == 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "amp circ buff has zero size" } );
            return;
        }

        std::cerr << "waiting to grow\n";
        while( m_ampCircBuff.size() < m_ampCircBuff.maxEntries() &&
               m_psdRestarting.load( std::memory_order_acquire ) == false && !shutdown() )
        {
            // shrinking sleep
            double stime = ( 1.0 * m_ampCircBuff.maxEntries() - 1.0 * m_ampCircBuff.size() ) / m_fps.load() * 0.5 * 1e9;
            mx::sys::nanoSleep( stime );
        }

        std::cerr << "all grown.  starting to calculate\n";

        meanSums.assign( m_nModes, 0 );
        meanHeadCache.clear();
        prevSnap         = ampCircBuffT::snapshotT();
        prevMeanRefEntry = 0;
        haveMeanSums     = false;
        avgPsdSum.clear();
        avgPsdWindowCount = 0;

        while( m_psdRestarting.load( std::memory_order_acquire ) == false && !shutdown() )
        {
            // Used to check if we are getting too behind
            uint64_t mono0 = m_ampCircBuff.mono();

            ampCircBuffT::snapshotT tsSnap;
            if( !loadPsdInputWindows( tsSnap ) )
            {
                continue;
            }

            cbIndexT tsRefEntry   = latestWindowRefEntry( tsSnap, m_tsSize );
            cbIndexT meanRefEntry = precedingWindowRefEntry( tsSnap, tsRefEntry, m_meanSize );

            cbIndexT meanCacheCount = std::min( m_meanSize, m_tsOverlapSize );
            if( meanCacheCount <= 0 )
            {
                meanCacheCount = 1;
            }

            bool canRollMean = false;
            if( haveMeanSums == true && prevSnap.maxEntries == tsSnap.maxEntries && meanHeadCache.size() > 0 )
            {
                cbIndexT advance = circularEntryAdvance( prevMeanRefEntry, meanRefEntry, tsSnap.maxEntries );
                canRollMean      = ( advance == meanCacheCount );

                if( canRollMean == true )
                {
                    rollMeanSums( meanSums, meanHeadCache, advance );
                }
            }

            if( canRollMean == false )
            {
                recomputeMeanSums( meanSums );
            }

            cacheMeanHead( meanHeadCache, meanCacheCount );
            prevSnap         = tsSnap;
            prevMeanRefEntry = meanRefEntry;
            haveMeanSums     = true;

            for( size_t m = 0; m < m_nModes; ++m ) // Loop over each mode
            {
                realT mn = static_cast<realT>( meanSums[m] / m_meanSize );

                double var = 0;

                for( cbIndexT n = 0; n < m_tsSize; ++n )
                {
                    m_tsWork[n] = m_tsPtrs[n][m] - mn; // load mean subtracted chunk

                    var += pow( m_tsWork[n], 2 );

                    m_tsWork[n] *= m_win[n];
                }
                var /= m_tsSize;

                m_fft( m_fftWork, m_tsWork );

                double nm = 0;
                for( size_t n = 0; n < m_psd.size(); ++n )
                {
                    m_psd[n] = norm( m_fftWork[n] );
                    nm += m_psd[n] * m_df;
                }

                // Put it in the buffer for uploading to shmim
                for( size_t n = 0; n < m_psd.size(); ++n )
                {
                    //                    m_psd[n] *= ( var / nm );
                    m_psdBuffer( n, m ) = m_psd[n] * ( var / nm );
                }
            }

            //------------------------- the raw psds ---------------------------
            m_rawpsdStream->md->write = 1;

            // Set the time of last write
            clock_gettime( CLOCK_REALTIME, &m_rawpsdStream->md->writetime );
            m_rawpsdStream->md->atime = m_rawpsdStream->md->writetime;

            uint64_t cnt1 = m_rawpsdStream->md->cnt1 + 1;
            if( cnt1 >= m_rawpsdStream->md->size[2] )
            {
                cnt1 = 0;
            }

            // Move to next pointer
            float *F = m_rawpsdStream->array.F + m_psdBuffer.rows() * m_psdBuffer.cols() * cnt1;

            if( m_rawPSDHistoryDepth > 0 && m_rawpsdStream->md->cnt0 >= m_rawpsdStream->md->size[2] )
            {
                size_t planeElements = m_psdBuffer.rows() * m_psdBuffer.cols();
                realT *H             = m_rawPSDHistory.data() + planeElements * m_rawPSDHistoryNext;

                memcpy( H, F, planeElements * sizeof( realT ) );

                ++m_rawPSDHistoryCount;
                ++m_rawPSDHistoryNext;
                if( m_rawPSDHistoryNext >= m_rawPSDHistoryDepth )
                {
                    m_rawPSDHistoryNext = 0;
                }
            }

            memcpy( F, m_psdBuffer.data(), m_psdBuffer.rows() * m_psdBuffer.cols() * sizeof( float ) );

            // Update cnt1
            m_rawpsdStream->md->cnt1 = cnt1;

            // Update cnt0
            ++m_rawpsdStream->md->cnt0;

            m_rawpsdStream->md->write = 0;
            ImageStreamIO_sempost( m_rawpsdStream, -1 );

            //-------------------------- now average the psds ----------------------------

            size_t planeElements = m_psdBuffer.rows() * m_psdBuffer.cols();
            if( avgPsdSum.size() != planeElements )
            {
                avgPsdSum.assign( planeElements, 0 );
                avgPsdWindowCount = 0;
            }

            uint64_t desiredPsdWindow  = static_cast<uint64_t>( desiredPSDAverageCount() );
            uint64_t storedRawPsdCount = storedRawPSDCount();
            uint64_t nextWindowCount   = std::min<uint64_t>( desiredPsdWindow, storedRawPsdCount );

            const realT *latestPlane = rawPSDPlaneByAge( 0, planeElements );

            bool recomputeAvgPsd = ( nextWindowCount == 0 || latestPlane == nullptr || avgPsdWindowCount == 0 ||
                                     nextWindowCount < avgPsdWindowCount || nextWindowCount > avgPsdWindowCount + 1 );

            if( recomputeAvgPsd == true )
            {
                recomputeAveragedPSDSum( avgPsdSum, nextWindowCount, planeElements );
            }
            else if( nextWindowCount == avgPsdWindowCount + 1 )
            {
                updatePlaneSum( avgPsdSum, latestPlane, nullptr, planeElements );
            }
            else
            {
                const realT *outgoingPlane = rawPSDPlaneByAge( nextWindowCount, planeElements );
                if( outgoingPlane == nullptr )
                {
                    recomputeAveragedPSDSum( avgPsdSum, nextWindowCount, planeElements );
                }
                else
                {
                    updatePlaneSum( avgPsdSum, latestPlane, outgoingPlane, planeElements );
                }
            }

            avgPsdWindowCount      = nextWindowCount;
            uint64_t avgPsdDivisor = ( avgPsdWindowCount == 0 ) ? 1 : avgPsdWindowCount;

            for( size_t n = 0; n < planeElements; ++n )
            {
                m_psdBuffer.data()[n] = static_cast<realT>( avgPsdSum[n] / avgPsdDivisor );
            }

            m_avgpsdStream->md->write = 1;

            // Set the time of last write
            clock_gettime( CLOCK_REALTIME, &m_avgpsdStream->md->writetime );
            m_avgpsdStream->md->atime = m_avgpsdStream->md->writetime;

            // Move to next pointer
            F = m_avgpsdStream->array.F;

            memcpy( F, m_psdBuffer.data(), m_psdBuffer.rows() * m_psdBuffer.cols() * sizeof( float ) );

            // Update cnt1
            m_avgpsdStream->md->cnt1 = 0;

            // Update cnt0
            ++m_avgpsdStream->md->cnt0;

            m_avgpsdStream->md->write = 0;
            ImageStreamIO_sempost( m_avgpsdStream, -1 );

            // double t1 = mx::sys::get_curr_time();
            // std::cerr << "done " << t1 - t0 << "\n";

            // Have to be cycling within the overlap
            if( m_ampCircBuff.mono() - mono0 >= static_cast<uint32_t>( m_tsOverlapSize ) )
            {
                log<text_log>( "PSD calculations getting behind, skipping ahead.", logPrio::LOG_WARNING );
            }
            else
            {
                while( m_ampCircBuff.mono() - mono0 < static_cast<uint32_t>( m_tsOverlapSize ) &&
                       !m_psdRestarting.load( std::memory_order_acquire ) )
                {
                    mx::sys::microSleep( 0.2 * 1000000.0 / m_fps.load( std::memory_order_acquire ) );
                }
            }

            if( m_psdRestarting.load( std::memory_order_acquire ) )
            {
                continue;
            }
        }
    }
}

modalPSDs::cbIndexT modalPSDs::latestWindowRefEntry( const ampCircBuffT::snapshotT &sn, cbIndexT count )
{
    if( sn.latest >= count )
    {
        return sn.latest - count;
    }

    return sn.maxEntries + sn.latest - count;
}

modalPSDs::cbIndexT
modalPSDs::precedingWindowRefEntry( const ampCircBuffT::snapshotT &sn, cbIndexT refEntry, cbIndexT count )
{
    if( refEntry >= count )
    {
        return refEntry - count;
    }

    return sn.maxEntries + refEntry - count;
}

modalPSDs::cbIndexT modalPSDs::circularEntryAdvance( cbIndexT from, cbIndexT to, cbIndexT maxEntries )
{
    if( maxEntries <= 0 )
    {
        return 0;
    }

    if( to >= from )
    {
        return to - from;
    }

    return maxEntries + to - from;
}

bool modalPSDs::loadPsdInputWindows( ampCircBuffT::snapshotT &sn )
{
    for( int retry = 0; retry < 3; ++retry )
    {
        ampCircBuffT::snapshotT seedSnap = m_ampCircBuff.snapshot();
        if( seedSnap.validEntries < m_ampCircBuff.maxEntries() )
        {
            return false;
        }

        cbIndexT tsRefEntry = latestWindowRefEntry( seedSnap, m_tsSize );

        if( !m_ampCircBuff.loadSequence( tsRefEntry, m_tsSize, m_tsPtrs.data(), sn ) )
        {
            return false;
        }

        if( sn.mono != seedSnap.mono )
        {
            continue;
        }

        cbIndexT meanRefEntry = precedingWindowRefEntry( sn, tsRefEntry, m_meanSize );

        ampCircBuffT::snapshotT meanSnap;
        if( !m_ampCircBuff.loadSequence( meanRefEntry, m_meanSize, m_meanPtrs.data(), meanSnap ) )
        {
            return false;
        }

        if( meanSnap.mono == sn.mono )
        {
            return true;
        }
    }

    return false;
}

void modalPSDs::recomputeMeanSums( std::vector<double> &meanSums ) const
{
    meanSums.assign( m_nModes, 0 );

    for( cbIndexT n = 0; n < m_meanSize; ++n )
    {
        const realT *sample = m_meanPtrs[n];
        for( size_t m = 0; m < m_nModes; ++m )
        {
            meanSums[m] += sample[m];
        }
    }
}

void modalPSDs::rollMeanSums( std::vector<double>      &meanSums,
                              const std::vector<realT> &meanHeadCache,
                              cbIndexT                  advance ) const
{
    if( advance <= 0 )
    {
        return;
    }

    for( cbIndexT n = 0; n < advance; ++n )
    {
        const realT *oldSample = meanHeadCache.data() + static_cast<size_t>( n ) * m_nModes;
        const realT *newSample = m_meanPtrs[m_meanSize - advance + n];

        for( size_t m = 0; m < m_nModes; ++m )
        {
            meanSums[m] += newSample[m] - oldSample[m];
        }
    }
}

void modalPSDs::cacheMeanHead( std::vector<realT> &meanHeadCache, cbIndexT count ) const
{
    if( count <= 0 )
    {
        meanHeadCache.clear();
        return;
    }

    meanHeadCache.resize( static_cast<size_t>( count ) * m_nModes );

    for( cbIndexT n = 0; n < count; ++n )
    {
        memcpy( meanHeadCache.data() + static_cast<size_t>( n ) * m_nModes, m_meanPtrs[n], m_nModes * sizeof( realT ) );
    }
}

bool modalPSDs::acceptLoopStateFrame() const
{
    return ( m_useLoopState == false ) || m_loopClosed.load( std::memory_order_acquire );
}

uint64_t modalPSDs::storedRawPSDCount() const
{
    return std::min<uint64_t>( m_rawpsdStream->md->cnt0, publishedRawPSDHistoryDepth() + m_rawPSDHistoryDepth );
}

const modalPSDs::realT *modalPSDs::rawPSDPlaneByAge( uint64_t age, size_t planeElements ) const
{
    const uint64_t publishedCount = std::min<uint64_t>( m_rawpsdStream->md->cnt0, publishedRawPSDHistoryDepth() );

    if( age < publishedCount )
    {
        uint64_t publishedDepth = m_rawpsdStream->md->size[2];
        uint64_t slot           = ( m_rawpsdStream->md->cnt1 + publishedDepth - age ) % publishedDepth;
        return m_rawpsdStream->array.F + planeElements * slot;
    }

    uint64_t overflowAge   = age - publishedCount;
    uint64_t overflowCount = std::min<uint64_t>( m_rawPSDHistoryCount, m_rawPSDHistoryDepth );
    if( overflowAge >= overflowCount || m_rawPSDHistoryDepth == 0 )
    {
        return nullptr;
    }

    uint64_t slot = ( m_rawPSDHistoryNext + m_rawPSDHistoryDepth - 1 - overflowAge ) % m_rawPSDHistoryDepth;
    return m_rawPSDHistory.data() + planeElements * slot;
}

void modalPSDs::recomputeAveragedPSDSum( std::vector<double> &avgPsdSum,
                                         uint64_t             windowCount,
                                         size_t               planeElements ) const
{
    avgPsdSum.assign( planeElements, 0 );

    for( uint64_t age = 0; age < windowCount; ++age )
    {
        const realT *plane = rawPSDPlaneByAge( age, planeElements );
        if( plane == nullptr )
        {
            break;
        }

        updatePlaneSum( avgPsdSum, plane, nullptr, planeElements );
    }
}

void modalPSDs::updatePlaneSum( std::vector<double> &planeSum,
                                const realT         *addPlane,
                                const realT         *removePlane,
                                size_t               planeElements )
{
    for( size_t n = 0; n < planeElements; ++n )
    {
        if( addPlane != nullptr )
        {
            planeSum[n] += addPlane[n];
        }

        if( removePlane != nullptr )
        {
            planeSum[n] -= removePlane[n];
        }
    }
}

int modalPSDs::desiredPSDAverageCount() const
{
    realT psdTime    = m_psdTime.load( std::memory_order_acquire );
    realT psdAvgTime = m_psdAvgTime.load( std::memory_order_acquire );

    if( psdTime <= 0 || m_psdOverlapFraction <= 0 )
    {
        return 1;
    }

    int nPSDAverage = static_cast<int>( std::ceil( psdAvgTime / ( psdTime * m_psdOverlapFraction ) ) );

    if( nPSDAverage <= 0 )
    {
        return 1;
    }

    return nPSDAverage;
}

modalPSDs::cbIndexT modalPSDs::desiredMeanSampleCount( realT fps ) const
{
    realT meanTime = m_meanTime.load( std::memory_order_acquire );

    if( fps <= 0 || meanTime <= 0 )
    {
        return 1;
    }

    cbIndexT meanSize = fps * meanTime;

    if( meanSize <= 0 )
    {
        return 1;
    }

    return meanSize;
}

modalPSDs::cbIndexT modalPSDs::requiredInputHistoryDepth() const
{
    if( m_tsSize <= 0 || m_meanSize <= 0 )
    {
        return 0;
    }

    // Leave one slot for the excluded latest sample and one for the unreadable overwrite edge.
    return m_tsSize + m_meanSize + 2;
}

uint32_t modalPSDs::rawPSDHistoryDepth() const
{
    const uint32_t desired   = static_cast<uint32_t>( desiredPSDAverageCount() );
    const uint32_t published = publishedRawPSDHistoryDepth();

    if( desired + 1 <= published )
    {
        return 0;
    }

    return desired + 1 - published;
}

uint32_t modalPSDs::publishedRawPSDHistoryDepth() const
{
    return std::max<uint32_t>( 1, static_cast<uint32_t>( m_nPSDHistory ) );
}

INDI_NEWCALLBACK_DEFN( modalPSDs, m_indiP_psdTime )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_psdTime, ipRecv );

    realT target;

    if( indiTargetUpdate( m_indiP_psdTime, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( m_psdTime.load( std::memory_order_acquire ) != target )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        m_psdTime.store( target, std::memory_order_release );

        updateIfChanged( m_indiP_psdTime, "current", m_psdTime.load(), INDI_IDLE );
        updateIfChanged( m_indiP_psdTime, "target", m_psdTime.load(), INDI_IDLE );

        shmimMonitorT::m_restart = true;

        log<text_log>( "set psdTime to " + std::to_string( m_psdTime.load() ), logPrio::LOG_NOTICE );
    }

    return 0;
} // INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_psdTime)

INDI_NEWCALLBACK_DEFN( modalPSDs, m_indiP_psdAvgTime )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_psdAvgTime, ipRecv );

    realT target;

    if( indiTargetUpdate( m_indiP_psdAvgTime, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( m_psdAvgTime.load( std::memory_order_acquire ) != target )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        m_psdAvgTime.store( target, std::memory_order_release );

        updateIfChanged( m_indiP_psdAvgTime, "current", m_psdAvgTime.load(), INDI_IDLE );
        updateIfChanged( m_indiP_psdAvgTime, "target", m_psdAvgTime.load(), INDI_IDLE );

        shmimMonitorT::m_restart = true;

        log<text_log>( "set psdAvgTime to " + std::to_string( m_psdAvgTime.load() ), logPrio::LOG_NOTICE );
    }

    return 0;
} // INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_psdAvgTime)

INDI_NEWCALLBACK_DEFN( modalPSDs, m_indiP_meanTime )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_meanTime, ipRecv );

    realT target;

    if( indiTargetUpdate( m_indiP_meanTime, target, ipRecv, true ) < 0 )
    {
        log<software_error>( { __FILE__, __LINE__ } );
        return -1;
    }

    if( m_meanTime.load( std::memory_order_acquire ) != target )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        m_meanTime.store( target, std::memory_order_release );

        updateIfChanged( m_indiP_meanTime, "current", m_meanTime.load(), INDI_IDLE );
        updateIfChanged( m_indiP_meanTime, "target", m_meanTime.load(), INDI_IDLE );

        shmimMonitorT::m_restart = true;

        log<text_log>( "set meanTime to " + std::to_string( m_meanTime.load() ), logPrio::LOG_NOTICE );
    }

    return 0;
} // INDI_NEWCALLBACK_DEFN(modalPSDs, m_indiP_meanTime)

INDI_SETCALLBACK_DEFN( modalPSDs, m_indiP_fpsSource )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_fpsSource, ipRecv );

    if( ipRecv.find( m_fpsElement ) != true ) // this isn't valid
    {
        log<software_error>( { __FILE__, __LINE__, "No current property in fps source." } );
        return 0;
    }

    std::lock_guard<std::mutex> guard( m_indiMutex );

    realT fps = ipRecv[m_fpsElement].get<realT>();

    if( fabs( fps - m_fps.load( std::memory_order_acquire ) ) > m_fpsTol + .0001 )
    {
        m_fps.store( fps, std::memory_order_release );
        log<text_log>( "set fps to " + std::to_string( m_fps.load() ), logPrio::LOG_NOTICE );
        updateIfChanged( m_indiP_fps, "current", m_fps.load(), INDI_IDLE );

        shmimMonitorT::m_restart = true;
    }

    return 0;

} // INDI_SETCALLBACK_DEFN(modalPSDs, m_indiP_fpsSource)

INDI_SETCALLBACK_DEFN( modalPSDs, m_indiP_loop )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_loop, ipRecv );

    if( ipRecv.find( m_loopStateElement ) != true )
    {
        log<software_error>( { __FILE__, __LINE__, "No configured loop-state element in loop source." } );
        return 0;
    }

    bool loopClosed = ( ipRecv[m_loopStateElement].getSwitchState() == pcf::IndiElement::On );

    if( loopClosed != m_loopClosed.load( std::memory_order_acquire ) )
    {
        std::lock_guard<std::mutex> guard( m_indiMutex );

        m_loopClosed.store( loopClosed, std::memory_order_release );
        shmimMonitorT::m_restart = true;

        log<text_log>( std::string( "loop state is now " ) + ( loopClosed ? "closed" : "open" ), logPrio::LOG_NOTICE );
    }

    return 0;

} // INDI_SETCALLBACK_DEFN(modalPSDs, m_indiP_loop)

} // namespace app
} // namespace MagAOX

#endif // modalPSDs_hpp
