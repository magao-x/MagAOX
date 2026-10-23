/** \file streamWriter.hpp
 * \brief The MagAO-X Image Stream Writer
 *
 * \ingroup streamWriter_files
 */

#ifndef streamWriter_hpp
#define streamWriter_hpp

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>

#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <xrif/xrif.h>

#include <mx/sys/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#define NOT_WRITING ( 0 )
#define START_WRITING ( 1 )
#define WRITING ( 2 )
#define STOP_WRITING ( 3 )

// #define SW_DEBUG

namespace MagAOX
{
namespace app
{

/** \defgroup streamWriter ImageStreamIO Stream Writing
 *  \brief Writes the contents of an ImageStreamIO image stream to disk.
 *
 *  <a href="../handbook/operating/software/apps/streamWriter.html">Application Documentation</a>
 *
 *  \ingroup apps
 *
 */

/** \defgroup streamWriter_files ImageStreamIO Stream Writing
 * \ingroup streamWriter
 */

/** MagAO-X application to control writing ImageStreamIO streams to disk.
 *
 * \ingroup streamWriter
 *
 */
class streamWriter : public MagAOXApp<>, public dev::telemeter<streamWriter>
{
    typedef dev::telemeter<streamWriter> telemeterT;

    friend class dev::telemeter<streamWriter>;

    // Give the test harness access.
    friend class streamWriter_test;
    friend class streamWriter_data_test;

  protected:
    /** \name Configurable Parameters - Data
     *@{
     */

    std::string m_rawimageDir; ///< The path where files will be saved.

    size_t m_maxCircBuffLength{ 1024 }; ///< The maximum length of the circular buffer, in frames

    double m_maxCircBuffSize{ 2048 }; ///< The maximum size of the circular buffer in MB.

    size_t m_maxWriteChunkLength{ 512 }; /**< The maximum number of frames to write at a time.  Must
                                              be an integer factor of m_maxCircBuffLength.*/

    double m_maxChunkTime{ 10 }; ///< The maximum time before writing regardless of number of frames.

    double m_writeStopTimeout{
        1.0 }; ///< Seconds to wait after a stop-writing command before flushing the pending data without a new frame.

    bool m_startWriting{ false }; ///< Whether writing should be armed automatically at application startup.

    std::string m_shmimName; ///< The name of the shared memory buffer.

    std::string m_outName; ///< The name to use for outputting files,  Default is m_shmimName.

    int m_semaphoreNumber{ 7 }; ///< The image structure semaphore index.

    unsigned m_semWaitSec{ 0 }; /**< The time in whole sec to wait on the semaphore,
                                     to which m_semWaitNSec is added.  Default is 0 sec.*/

    unsigned m_semWaitNSec{ 500000000 }; /**< The time in nsec to wait on the semaphore, added to m_semWaitSec.
                                              Max is 999999999. Default is 5e8 nsec. */

    bool m_warnMissedData{
        true }; ///< Whether missed-data backlog summaries should be emitted as warnings instead of informational logs.

    int m_lz4accel{ 1 }; ///< LZ4 acceleration used for each image archive.

    bool m_compress{ true }; ///< Whether image archives use XRIF compression.

    ///@}

    size_t m_circBuffLength{ 1024 };  ///< The length of the circular buffer, in frames
    double m_circBuffSize{ 2048.0 };  ///< The size of the circular buffer, in MB
    size_t m_writeChunkLength{ 512 }; ///< The number of frames to write at a time

    size_t  m_width{ 0 };    ///< The width of the image
    size_t  m_height{ 0 };   ///< The height of the image
    uint8_t m_dataType{ 0 }; ///< The ImageStreamIO type code.
    int     m_typeSize{ 0 }; ///< The pixel byte depth

    char     *m_rawImageCircBuff{ nullptr }; ///< Owned pixel storage for the ingest circular buffer.
    uint64_t *m_timingCircBuff{ nullptr };   ///< Owned five-word timing records for each buffered frame.

    size_t m_currImage{ 0 }; ///< Next destination frame in the circular buffer.

    uint64_t m_currImageTime{ 0 }; ///< The write-time of the current image in nanoseconds.

    uint64_t m_currChunkStartTime{ 0 }; ///< The write-time of the first image in the chunk in nanoseconds.

    std::atomic<uint64_t> m_skippedFrameCount{
        0 }; ///< Count of skipped frames accumulated by the framegrabber thread since the last summary log.

    std::atomic<uint64_t> m_repeatSemaphoreCount{
        0 }; ///< Count of repeated semaphore wakes with unchanged frame count since the last summary log.

    double m_skipSummaryIntervalSec{ 10.0 }; ///< Current interval between summary skip logs.

    double m_nextSkipSummaryTime{ 0.0 }; ///< Time after which the next summary skip log may be emitted.

    // Writer book-keeping:
    std::atomic<int> m_writing{ NOT_WRITING }; ///< Published state; transitions are serialized by m_writeMutex.

    std::mutex m_writeMutex; ///< Protects capture state and the single writer-owned save descriptor.

    std::condition_variable m_writeCond; ///< Wakes the producer when the writer releases its save window.

    size_t m_chunkFrames{ 0 }; ///< Unsaved frames in the current contiguous capture chunk, protected by m_writeMutex.

    bool m_stopQueued{ false }; ///< The pending save completes the stop; no more recording frames may be appended.

    std::atomic<bool> m_fgReady{ false }; ///< The source has completed allocation, semaphore flushing, and baselining.

    std::atomic<bool> m_fgStopped{ true }; ///< The producer can no longer publish work during shutdown.

    std::atomic<bool> m_writerFailed{ false }; ///< Failed writer completion prevents any further publication.

    std::atomic<bool> m_writePending{
        false }; ///< Whether the writer thread still owns a queued save window and may touch the circular buffers.

    double m_stopWriteDeadline{ 0.0 }; ///< Absolute time after which a stop-writing request should flush without a
                                       ///< new frame.

    double m_writeCompletionTimeout{
        5.0 }; ///< Maximum producer wait for a queued save before buffer reuse, publication, or reconnect cleanup.

    bool m_resumeAfterReconnect{
        false }; ///< Tracks whether reconnect cleanup should resume writing immediately on the replacement stream.

    uint64_t m_currChunkStart{ 0 }; ///< First unsaved frame in the current contiguous chunk, protected by m_writeMutex.
    uint64_t m_nextChunkStart{ 0 }; ///< Aligned chunk boundary used to limit the next save window.

    uint64_t m_currSaveStart{ 0 }; ///< The circular buffer position at which to start saving.
    uint64_t m_currSaveStop{ 0 };  ///< The circular buffer position at which to stop saving.

    std::atomic<uint64_t> m_currSaveStopFrameNo{
        0 }; ///< Source-frame counter at the end of the latest queued save window.

    double m_compressionRatio{ 0 }; ///< Last completed encoder ratio, protected by m_writeMutex.

    double m_encodeRate{ 0 }; ///< Last completed image encoder byte rate, protected by m_writeMutex.

    double m_differenceRate{ 0 }; ///< Last completed differencing byte rate, protected by m_writeMutex.

    double m_reorderRate{ 0 }; ///< Last completed reorder byte rate, protected by m_writeMutex.

    double m_compressRate{ 0 }; ///< Last completed compression byte rate, protected by m_writeMutex.

    size_t m_frameBytes{ 0 }; ///< Frame size associated with the published encoder rates, protected by m_writeMutex.

    int16_t m_lastSavingState{ -1 }; ///< Last saving-state telemetry value, protected by m_writeMutex.

    uint64_t m_lastSavingFrame{ UINT64_MAX }; ///< Last saving-state telemetry frame, protected by m_writeMutex.

    /// The xrif compression handle for image data
    xrif_t m_xrif{ nullptr };

    /// Storage for the xrif image data file header
    char *m_xrif_header{ nullptr };

    /// The owned uncompressed XRIF encoder for frame timing records.
    xrif_t m_xrif_timing{ nullptr };

    /// Owned archive header for the timing encoder.
    char *m_xrif_timing_header{ nullptr };

    std::string m_outFilePath; ///< The full path for the latest output file

  public:
    /// Construct an idle writer with power management disabled.
    streamWriter();

    /// Release owned circular buffers, encoders, and archive headers.
    ~streamWriter() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// Load configuration and request shutdown if a helper fails.
    virtual void loadConfig();

    /// Load configuration while reporting helper failures.
    /** \returns 0 on success, or -1 for invalid semaphore waits or failed telemetry configuration. */
    int loadConfigImpl( mx::app::appConfigurator &config /**< [in/out] application configuration */ );

    /// Startup functions
    /** Sets up the INDI vars.
     *
     */
    virtual int appStartup();

    /// Monitor workers, publish writing state, and schedule telemetry in every state.
    virtual int appLogic();

    /// Join worker threads, release buffers and encoders, and stop telemetry.
    virtual int appShutdown();

  protected:
    std::atomic<bool> m_restart{ false }; ///< Requests reopening the source stream after metadata or inode changes.

    /// Initialize the xrif system.
    /** Allocates the handles and headers pointers.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int initialize_xrif();

    /** \name Framegrabber Thread
     * This thread monitors the ImageStreamIO buffer and copies its images to the circular buffer.
     *
     * @{
     */
    int m_fgThreadPrio{ 1 }; ///< Priority of the framegrabber thread, should normally be > 0.

    std::string m_fgCpuset; ///< The cpuset for the framegrabber thread.  Ignored if empty (the default).

    std::thread m_fgThread; ///< Worker owning source acquisition and circular-buffer publication.

    bool m_fgThreadInit{ true }; ///< Synchronizer to ensure f.g. thread initializes before doing dangerous things.

    pid_t m_fgThreadID{ 0 }; ///< F.g. thread PID.

    pcf::IndiProperty m_fgThreadProp; ///< The property to hold the f.g. thread details.

  public:
    /// Select circular-buffer and chunk sizes within the configured frame and memory limits.
    static void getCircBuffLengths( size_t  &circBuffLength /**< [out] selected buffer length in frames */,
                                    double  &circBuffSize /**< [out] selected buffer size in MiB */,
                                    size_t  &writeChunkLength /**< [out] selected write chunk length */,
                                    size_t   maxCircBuffLength /**< [in] maximum buffer length */,
                                    double   maxCircBuffSize /**< [in] maximum buffer size in MiB */,
                                    size_t   maxWriteChunkLength /**< [in] maximum write chunk length */,
                                    uint32_t width /**< [in] frame width in pixels */,
                                    uint32_t height /**< [in] frame height in pixels */,
                                    size_t   typeSize /**< [in] pixel size in bytes */ );

  protected:
    /// Worker function to allocate the circular buffers.
    /** This takes place in the fg thread after connecting to the stream.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int allocate_circbufs();

    /// Worker function to configure and allocate the xrif handles.
    /** This takes place in the fg thread after connecting to the stream.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int allocate_xrif();

    /// Release the circular buffers owned by the framegrabber thread.
    void release_circbufs();

    /// Wait for successful writer completion before reusing or releasing its save window.
    /** \returns false on writer failure or expiration of m_writeCompletionTimeout. */
    bool waitForWriteCompletion( uint64_t saveStopFrameNo /**< [in] The last frame number associated with the
                                                               queued save for timeout logging. */ );

    /// Publish the current chunk once the preceding descriptor has been released.
    /** The caller retains its unsaved chunk while waiting, and callbacks may change stop intent during the wait.
     * \returns 0 on success, or -1 on writer failure, a timeout, an invalid window, or a semaphore error. */
    int queueWrite( std::unique_lock<std::mutex> &lock /**< [in/out] held capture-state lock, released while waiting */,
                    uint64_t                      frameNo /**< [in] final source-frame counter in the chunk */ );

    /// Release the queued window and complete its stop transition after encoding.
    void completeWrite( bool success = true /**< [in] whether encoding completed successfully */ );

    /// Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
    static void fgThreadStart( streamWriter *s /**< [in] a pointer to an streamWriter instance (normally this) */ );

    /// Execute the frame grabber main loop.
    void fgThreadExec();

    /// Run source acquisition while fgThreadExec publishes worker lifetime and readiness.
    void fgThreadRun();

    ///@}

    /** \name Stream Writer Thread
     * This thread writes chunks of the circular buffer to disk.
     *
     * @{
     */
    int m_swThreadPrio{ 1 }; ///< Priority of the stream writer thread, should normally be > 0, and <= m_fgThreadPrio.

    std::string m_swCpuset; ///< The cpuset for the stream writer thread.  Ignored if empty (the default).

    sem_t m_swSemaphore; ///< Semaphore used to synchronize the fg thread and the sw thread.

    std::thread m_swThread; ///< A separate thread for the actual writing

    bool m_swThreadInit{ true }; ///< Synchronizer to ensure s.w. thread initializes before doing dangerous things.

    pid_t m_swThreadID{ 0 }; ///< S.w. thread pid.

    pcf::IndiProperty m_swThreadProp; ///< The property to hold the s.w. thread details.

    /// Thread starter, called by swThreadStart on thread construction.  Calls swThreadExec.
    static void swThreadStart( streamWriter *s /**< [in] a pointer to an streamWriter instance (normally this) */ );

    /// Execute the stream writer main loop.
    void swThreadExec();

    /// Validate and encode a stable save window, releasing its ownership on completion or failure.
    /** The writer calls this only for a pending notification; direct callers must provide a complete descriptor.
     * \returns 0 on success, or -1 on invalid input or a fatal encoding/output prerequisite failure. */
    int doEncode();
    ///@}

    // INDI:
  protected:
    // declare our properties
    pcf::IndiProperty m_indiP_writing; ///< Writing request and current toggle state.

    pcf::IndiProperty m_indiP_xrifStats; ///< Published image compression ratio and rates.

  public:
    /// Validate and apply a writing toggle request.
    INDI_NEWCALLBACK_DECL( streamWriter, m_indiP_writing );

    /// Publish the writing toggle and current encoder performance.
    void updateINDI();

    /** \name Telemeter Interface
     *
     * @{
     */
    /// Check whether the saving-state telemetry record is due.
    int checkRecordTimes();

    /// Force a scheduled saving-state telemetry record.
    int recordTelem( const telem_saving_state *record /**< [in] telemetry type selector */ );

    /// Record saving activity and the source-frame counter at the end of the latest queued save window.
    int recordSavingState( bool force = false /**< [in] emit even without a state change */ );

    /// Record changes in encoder sizes and rates.
    int recordSavingStats( bool force = false /**< [in] emit even without a statistics change */ );

    ///@}
};

streamWriter::streamWriter() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_powerMgtEnabled = false;

    return;
}

streamWriter::~streamWriter() noexcept
{
    release_circbufs();

    if( m_xrif )
        xrif_delete( m_xrif );

    if( m_xrif_header )
        free( m_xrif_header );

    if( m_xrif_timing )
        xrif_delete( m_xrif_timing );

    if( m_xrif_timing_header )
        free( m_xrif_timing_header );

    return;
}

void streamWriter::setupConfig()
{
    config.add( "writer.savePath",
                "",
                "writer.savePath",
                argType::Required,
                "writer",
                "savePath",
                false,
                "string",
                "The absolute path where images are saved. Will use MagAO-X default if not set." );

    config.add( "writer.maxCircBuffLength",
                "",
                "writer.maxCircBuffLength",
                argType::Required,
                "writer",
                "maxCircBuffLength",
                false,
                "size_t",
                "The maximum length in frames of the circular buffer. Should be an integer multiple of and larger than "
                "maxWriteChunkLength." );

    config.add( "writer.maxCircBuffSize",
                "",
                "writer.maxCircBuffSize",
                argType::Required,
                "writer",
                "maxCircBuffSize",
                false,
                "double",
                "The maximum size in MB of the circular buffer. Should be sized to hold at least 2 of the maximum "
                "frame size." );

    config.add(
        "writer.maxWriteChunkLength",
        "",
        "writer.maxWriteChunkLength",
        argType::Required,
        "writer",
        "maxWriteChunkLength",
        false,
        "size_t",
        "The maximum length in frames of the chunks to write to disk. Should be smaller than maxCircBuffLength." );

    config.add( "writer.maxChunkTime",
                "",
                "writer.maxChunkTime",
                argType::Required,
                "writer",
                "maxChunkTime",
                false,
                "float",
                "The max length in seconds of the chunks to write to disk. Default is 10 sec." );

    config.add( "writer.stopTimeout",
                "",
                "writer.stopTimeout",
                argType::Required,
                "writer",
                "stopTimeout",
                false,
                "float",
                "The max time in seconds to wait after a stop-writing command for the next frame before flushing the "
                "pending data and returning to the idle state." );

    config.add( "writer.startWriting",
                "",
                "writer.startWriting",
                argType::Required,
                "writer",
                "startWriting",
                false,
                "bool",
                "Flag controlling whether writing is armed automatically at application startup. Default is false." );

    config.add( "writer.threadPrio",
                "",
                "writer.threadPrio",
                argType::Required,
                "writer",
                "threadPrio",
                false,
                "int",
                "The real-time priority of the stream writer thread." );

    config.add( "writer.cpuset",
                "",
                "writer.cpuset",
                argType::Required,
                "writer",
                "cpuset",
                false,
                "int",
                "The cpuset for the writer thread." );

    config.add( "writer.compress",
                "",
                "writer.compress",
                argType::Required,
                "writer",
                "compress",
                false,
                "bool",
                "Flag to set whether compression is used.  Default true." );

    config.add( "writer.lz4accel",
                "",
                "writer.lz4accel",
                argType::Required,
                "writer",
                "lz4accel",
                false,
                "int",
                "The LZ4 acceleration parameter.  Larger is faster, but lower compression." );

    config.add( "writer.outName",
                "",
                "writer.outName",
                argType::Required,
                "writer",
                "outName",
                false,
                "int",
                "The name to use for output files.  Default is the shmimName." );

    config.add( "framegrabber.shmimName",
                "",
                "framegrabber.shmimName",
                argType::Required,
                "framegrabber",
                "shmimName",
                false,
                "int",
                "The name of the stream to monitor. From /tmp/shmimName.im.shm." );

    config.add( "framegrabber.semaphoreNumber",
                "",
                "framegrabber.semaphoreNumber",
                argType::Required,
                "framegrabber",
                "semaphoreNumber",
                false,
                "int",
                "The semaphore to wait on. Default is 7." );

    config.add( "framegrabber.semWait",
                "",
                "framegrabber.semWait",
                argType::Required,
                "framegrabber",
                "semWait",
                false,
                "int",
                "The time in nsec to wait on the semaphore, from 1 to 999999999. Default is 5e8 nsec." );

    config.add( "framegrabber.warnMissedData",
                "",
                "framegrabber.warnMissedData",
                argType::Required,
                "framegrabber",
                "warnMissedData",
                false,
                "bool",
                "Whether missed-data backlog summaries should be emitted at warning priority. Default is true." );

    config.add( "framegrabber.threadPrio",
                "",
                "framegrabber.threadPrio",
                argType::Required,
                "framegrabber",
                "threadPrio",
                false,
                "int",
                "The real-time priority of the framegrabber thread." );

    config.add( "framegrabber.cpuset",
                "",
                "framegrabber.cpuset",
                argType::Required,
                "framegrabber",
                "cpuset",
                false,
                "string",
                "The cpuset for the framegrabber thread." );

    TELEMETER_SETUP_CONFIG( config );
}

int streamWriter::loadConfigImpl( mx::app::appConfigurator &config )
{

    config( m_maxCircBuffLength, "writer.maxCircBuffLength" );
    config( m_maxCircBuffSize, "writer.maxCircBuffSize" );
    config( m_maxWriteChunkLength, "writer.maxWriteChunkLength" );
    config( m_maxChunkTime, "writer.maxChunkTime" );
    config( m_writeStopTimeout, "writer.stopTimeout" );
    if( m_writeStopTimeout < 0 )
    {
        m_writeStopTimeout = 0;
    }
    config( m_startWriting, "writer.startWriting" );
    config( m_swThreadPrio, "writer.threadPrio" );
    config( m_swCpuset, "writer.cpuset" );
    config( m_compress, "writer.compress" );
    config( m_lz4accel, "writer.lz4accel" );
    if( m_lz4accel < XRIF_LZ4_ACCEL_MIN )
    {
        m_lz4accel = XRIF_LZ4_ACCEL_MIN;
    }
    if( m_lz4accel > XRIF_LZ4_ACCEL_MAX )
    {
        m_lz4accel = XRIF_LZ4_ACCEL_MAX;
    }

    config( m_shmimName, "framegrabber.shmimName" );

    m_outName = m_shmimName;
    config( m_outName, "writer.outName" );

    config( m_semaphoreNumber, "framegrabber.semaphoreNumber" );
    config( m_semWaitNSec, "framegrabber.semWait" );
    if( m_semWaitNSec == 0 || m_semWaitNSec > 999999999 )
    {
        return log<software_error, -1>(
            { __FILE__, __LINE__, "framegrabber.semWait must be between 1 and 999999999 ns" } );
    }
    config( m_warnMissedData, "framegrabber.warnMissedData" );

    config( m_fgThreadPrio, "framegrabber.threadPrio" );
    config( m_fgCpuset, "framegrabber.cpuset" );

    // Set some defaults
    // Setup default saving path
    std::string tmpstr = mx::sys::getEnv( MAGAOX_env_rawimage );
    if( tmpstr == "" )
    {
        tmpstr = MAGAOX_rawimageRelPath;
    }
    m_rawimageDir = basePath() + "/" + tmpstr;

    config( m_rawimageDir, "writer.savePath" );

    TELEMETER_LOAD_CONFIG( config );

    return 0;
}

void streamWriter::loadConfig()
{
    if( loadConfigImpl( config ) < 0 )
    {
        m_shutdown = true;
    }
}

int streamWriter::appStartup()
{
    // Create save directory.
    errno = 0;
    if( mkdir( m_rawimageDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH ) < 0 )
    {
        if( errno != EEXIST )
        {
            std::stringstream logss;
            logss << "Failed to create image directory (" << m_rawimageDir << ").  Errno says: " << strerror( errno );
            log<software_critical>( { __FILE__, __LINE__, errno, 0, logss.str() } );

            return -1;
        }
    }

    // set up the  INDI properties
    createStandardIndiToggleSw( m_indiP_writing, "writing" );
    registerIndiPropertyNew( m_indiP_writing, INDI_NEWCALLBACK( m_indiP_writing ) );

    // Register the stats INDI property
    REG_INDI_NEWPROP_NOCB( m_indiP_xrifStats, "xrif", pcf::IndiProperty::Number );
    m_indiP_xrifStats.setLabel( "xrif compression performance" );

    indi::addNumberElement<float>( m_indiP_xrifStats, "ratio", 0, 1.0, 0.0, "%0.2f", "Compression Ratio" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "differenceMBsec",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Differencing Rate [MB/sec]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "reorderMBsec",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Reordering Rate [MB/sec]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "compressMBsec",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Compression Rate [MB/sec]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "encodeMBsec",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Total Encoding Rate [MB/sec]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "differenceFPS",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Differencing Rate [f.p.s.]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "reorderFPS",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Reordering Rate [f.p.s.]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "compressFPS",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Compression Rate [f.p.s.]" );

    indi::addNumberElement<float>( m_indiP_xrifStats,
                                   "encodeFPS",
                                   0,
                                   std::numeric_limits<float>::max(),
                                   0.0,
                                   "%0.2f",
                                   "Total Encoding Rate [f.p.s.]" );

    // Now set up the framegrabber and writer threads.
    //  - initialize the semaphore
    //  - start the threads

    if( sem_init( &m_swSemaphore, 0, 0 ) < 0 )
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Initializing S.W. semaphore" } );

    // Check if we have a safe writeChunkLengthh
    if( m_maxCircBuffLength % m_maxWriteChunkLength != 0 )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "Write chunk length is not a divisor of circular buffer length." } );
    }

    if( initialize_xrif() < 0 )
    {
        log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( m_startWriting )
    {
        m_writing = START_WRITING;
    }

    if( threadStart( m_fgThread,
                     m_fgThreadInit,
                     m_fgThreadID,
                     m_fgThreadProp,
                     m_fgThreadPrio,
                     m_fgCpuset,
                     "framegrabber",
                     this,
                     fgThreadStart ) < 0 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    if( threadStart( m_swThread,
                     m_swThreadInit,
                     m_swThreadID,
                     m_swThreadProp,
                     m_swThreadPrio,
                     m_swCpuset,
                     "streamwriter",
                     this,
                     swThreadStart ) < 0 )
    {
        log<software_critical, -1>( { __FILE__, __LINE__ } );
    }

    TELEMETER_APP_STARTUP;

    return 0;
}

int streamWriter::appLogic()
{
    double now = mx::sys::get_curr_time();
    if( m_nextSkipSummaryTime == 0 )
    {
        m_nextSkipSummaryTime = now + m_skipSummaryIntervalSec;
    }

    if( now >= m_nextSkipSummaryTime )
    {
        uint64_t skippedFrames   = m_skippedFrameCount.exchange( 0 );
        uint64_t repeatedSems    = m_repeatSemaphoreCount.exchange( 0 );
        double   summaryInterval = m_skipSummaryIntervalSec;

        bool shouldLogSkippedFrames = skippedFrames > 0;
        bool shouldLog              = shouldLogSkippedFrames || repeatedSems > 0;

        if( shouldLog )
        {
            std::string msg      = "stream ingest backlog: ";
            bool        appended = false;

            if( shouldLogSkippedFrames )
            {
                msg += std::to_string( skippedFrames ) + " skipped frames";
                appended = true;
            }

            if( repeatedSems > 0 )
            {
                if( appended )
                {
                    msg += ", ";
                }
                msg += std::to_string( repeatedSems ) + " repeated semaphore wakes";
            }
            msg += " in last " + std::to_string( static_cast<int>( summaryInterval ) ) + " sec";

            log<text_log>( msg, m_warnMissedData ? logPrio::LOG_WARNING : logPrio::LOG_INFO );

            m_skipSummaryIntervalSec *= 2.0;
            if( m_skipSummaryIntervalSec > 60.0 )
            {
                m_skipSummaryIntervalSec = 60.0;
            }
        }
        else
        {
            m_skipSummaryIntervalSec = 10.0;
        }

        m_nextSkipSummaryTime = now + m_skipSummaryIntervalSec;
    }

    // first do a join check to see if other threads have exited.
    // these will throw if the threads are really gone
    try
    {
        if( pthread_tryjoin_np( m_fgThread.native_handle(), 0 ) == 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "framegrabber thread has exited" } );
            return -1;
        }
    }
    catch( ... )
    {
        log<software_error>( { __FILE__, __LINE__, "framegrabber thread has exited" } );
        return -1;
    }

    try
    {
        if( pthread_tryjoin_np( m_swThread.native_handle(), 0 ) == 0 )
        {
            log<software_error>( { __FILE__, __LINE__, "stream thread has exited" } );
            return -1;
        }
    }
    catch( ... )
    {
        log<software_error>( { __FILE__, __LINE__, "streamwriter thread has exited" } );
        return -1;
    }

    switch( m_writing )
    {
    case NOT_WRITING:
        state( stateCodes::READY );
        break;
    default:
        state( stateCodes::OPERATING );
    }

    TELEMETER_APP_LOGIC;

    updateINDI();

    return 0;
}

int streamWriter::appShutdown()
{
    m_shutdown = 1;

    try
    {
        if( m_fgThread.joinable() )
        {
            m_fgThread.join();
        }
    }
    catch( ... )
    {
    }

    try
    {
        if( m_swThread.joinable() )
        {
            m_swThread.join();
        }
    }
    catch( ... )
    {
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_writeMutex );
        m_writing              = NOT_WRITING;
        m_writePending         = false;
        m_chunkFrames          = 0;
        m_stopQueued           = false;
        m_resumeAfterReconnect = false;
        m_stopWriteDeadline    = 0;
    }
    updateINDI();

    release_circbufs();

    if( m_xrif )
    {
        xrif_delete( m_xrif );
        m_xrif = nullptr;
    }

    if( m_xrif_timing )
    {
        xrif_delete( m_xrif_timing );
        m_xrif_timing = nullptr;
    }

    TELEMETER_APP_SHUTDOWN;

    return 0;
}

int streamWriter::initialize_xrif()
{
    xrif_error_t rv = xrif_new( &m_xrif );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, 0, rv, "xrif handle allocation or initialization error." } );
    }

    if( m_compress )
    {
        rv = xrif_configure( m_xrif, XRIF_DIFFERENCE_PREVIOUS, XRIF_REORDER_BYTEPACK, XRIF_COMPRESS_LZ4 );
        if( rv != XRIF_NOERROR )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif handle configuration error." } );
        }
    }
    else
    {
        std::cerr << "not compressing . . . \n";
        rv = xrif_configure( m_xrif, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE );
        if( rv != XRIF_NOERROR )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif handle configuration error." } );
        }
    }

    errno = 0;

    m_xrif_header = reinterpret_cast<char *>( malloc( XRIF_HEADER_SIZE * sizeof( char ) ) );
    if( m_xrif_header == NULL )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "xrif header allocation failed." } );
    }

    rv = xrif_new( &m_xrif_timing );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, 0, rv, "xrif handle allocation or initialization error." } );
    }

    // m_xrif_timing->reorder_method = XRIF_REORDER_NONE;
    rv = xrif_configure( m_xrif_timing, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif handle configuration error." } );
    }

    errno = 0;

    m_xrif_timing_header = reinterpret_cast<char *>( malloc( XRIF_HEADER_SIZE * sizeof( char ) ) );
    if( m_xrif_timing_header == NULL )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "xrif header allocation failed." } );
    }

    return 0;
}

void streamWriter::getCircBuffLengths( size_t  &circBuffLength,
                                       double  &circBuffSize,
                                       size_t  &writeChunkLength,
                                       size_t   maxCircBuffLength,
                                       double   maxCircBuffSize,
                                       size_t   maxWriteChunkLength,
                                       uint32_t width,
                                       uint32_t height,
                                       size_t   typeSize )
{
    static constexpr double MB = 1048576.0;

    size_t isz = width * height * typeSize * maxCircBuffLength;

    if( isz <= maxCircBuffSize * MB )
    {
        circBuffLength   = maxCircBuffLength;
        circBuffSize     = isz / MB;
        writeChunkLength = maxWriteChunkLength;

        return;
    }

    circBuffLength = maxCircBuffSize * MB / ( width * height * typeSize );

    if( circBuffLength % 2 == 1 )
    {
        --circBuffLength;
    }

    circBuffSize = ( width * height * typeSize * circBuffLength ) / MB;

    writeChunkLength = ( 1.0 * maxWriteChunkLength / maxCircBuffLength ) * circBuffLength;

    if( circBuffLength == 0 )
    {
        return;
    }

    if( writeChunkLength == 0 )
    {
        writeChunkLength = 1;
    }

    while( circBuffLength % writeChunkLength != 0 )
    {
        --writeChunkLength;
    }

    return;
}

int streamWriter::allocate_circbufs()
{

    getCircBuffLengths( m_circBuffLength,
                        m_circBuffSize,
                        m_writeChunkLength,
                        m_maxCircBuffLength,
                        m_maxCircBuffSize,
                        m_maxWriteChunkLength,
                        m_width,
                        m_height,
                        m_typeSize );

    if( m_circBuffLength < 2 )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "frame size too large to fit in maxCircBuffSize" } );
    }

    if( m_writeChunkLength >= m_circBuffLength )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "writeChunkLength is not smaller than circBuffLength" } );
    }

    if( m_circBuffLength % m_writeChunkLength != 0 )
    {
        return log<software_critical, -1>(
            { __FILE__, __LINE__, "writeChunkLength is not an integer factor of circBuffLength" } );
    }

    std::string msg = "Set circ buff length: " + std::to_string( m_circBuffLength ) + " frames (";
    msg += std::to_string( m_circBuffSize ) + " MB).  Write chunk length: " + std::to_string( m_writeChunkLength );
    msg += " frames.";

    log<text_log>( msg, logPrio::LOG_NOTICE );

    if( m_rawImageCircBuff )
    {
        free( m_rawImageCircBuff );
    }

    errno              = 0;
    m_rawImageCircBuff = reinterpret_cast<char *>( malloc( m_width * m_height * m_typeSize * m_circBuffLength ) );

    if( m_rawImageCircBuff == NULL )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "buffer allocation failure" } );
    }

    if( m_timingCircBuff )
    {
        free( m_timingCircBuff );
    }

    errno            = 0;
    m_timingCircBuff = reinterpret_cast<uint64_t *>( malloc( 5 * sizeof( uint64_t ) * m_circBuffLength ) );
    if( m_timingCircBuff == NULL )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "buffer allocation failure" } );
    }

    return 0;
}

int streamWriter::allocate_xrif()
{
    // Set up the image data xrif handle
    xrif_error_t rv;

    if( m_compress )
    {
        rv = xrif_configure( m_xrif, XRIF_DIFFERENCE_PREVIOUS, XRIF_REORDER_BYTEPACK, XRIF_COMPRESS_LZ4 );
        if( rv != XRIF_NOERROR )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif handle configuration error." } );
        }
    }
    else
    {
        std::cerr << "not compressing . . . \n";
        rv = xrif_configure( m_xrif, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE );
        if( rv != XRIF_NOERROR )
        {
            return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif handle configuration error." } );
        }
    }

    rv = xrif_set_size( m_xrif, m_width, m_height, 1, m_writeChunkLength, m_dataType );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif_set_size error." } );
    }

    rv = xrif_allocate_raw( m_xrif );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif_allocate_raw error." } );
    }

    rv = xrif_allocate_reordered( m_xrif );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif_allocate_reordered error." } );
    }

    // Set up the timing data xrif handle
    rv = xrif_configure( m_xrif_timing, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif handle configuration error." } );
    }

    rv = xrif_set_size( m_xrif_timing, 5, 1, 1, m_writeChunkLength, XRIF_TYPECODE_UINT64 );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif_set_size error." } );
    }

    rv = xrif_allocate_raw( m_xrif_timing );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif_allocate_raw error." } );
    }

    rv = xrif_allocate_reordered( m_xrif_timing );
    if( rv != XRIF_NOERROR )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, 0, rv, "xrif_allocate_reordered error." } );
    }

    return 0;
}

void streamWriter::release_circbufs()
{
    if( m_rawImageCircBuff )
    {
        free( m_rawImageCircBuff );
        m_rawImageCircBuff = nullptr;
    }

    if( m_timingCircBuff )
    {
        free( m_timingCircBuff );
        m_timingCircBuff = nullptr;
    }
}

bool streamWriter::waitForWriteCompletion( uint64_t saveStopFrameNo )
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>( m_writeCompletionTimeout );
    std::unique_lock<std::mutex> lock( m_writeMutex );
    if( m_writeCond.wait_until( lock, deadline, [this] { return !m_writePending || m_writerFailed; } ) )
    {
        return !m_writerFailed;
    }
    log<text_log>( "timed out waiting " + std::to_string( m_writeCompletionTimeout ) +
                       " sec for the writer thread to finish frame " + std::to_string( saveStopFrameNo ),
                   logPrio::LOG_NOTICE );
    return false;
}

int streamWriter::queueWrite( std::unique_lock<std::mutex> &lock, uint64_t frameNo )
{
    if( m_writerFailed )
    {
        return -1;
    }
    if( m_writePending )
    {
        const uint64_t previousFrame = m_currSaveStopFrameNo;
        lock.unlock();
        const bool completed = waitForWriteCompletion( previousFrame );
        lock.lock();
        if( !completed )
        {
            m_shutdown = 1;
            return -1;
        }
    }

    if( m_currChunkStart > m_circBuffLength || m_chunkFrames > m_circBuffLength - m_currChunkStart ||
        m_chunkFrames > m_writeChunkLength )
    {
        return log<software_critical, -1>( { __FILE__, __LINE__, "invalid capture chunk" } );
    }
    if( m_chunkFrames == 0 )
    {
        return 0;
    }

    m_currSaveStart       = m_currChunkStart;
    m_currSaveStop        = m_currChunkStart + m_chunkFrames;
    m_currSaveStopFrameNo = frameNo;
    m_stopQueued          = m_writing == STOP_WRITING;
    m_chunkFrames         = 0;
    m_currChunkStart      = m_currSaveStop == m_circBuffLength ? 0 : m_currSaveStop;
    m_nextChunkStart      = ( m_currChunkStart / m_writeChunkLength ) * m_writeChunkLength;
    m_stopWriteDeadline   = 0;
    m_writePending        = true;
    if( sem_post( &m_swSemaphore ) < 0 )
    {
        m_writePending = false;
        m_writeCond.notify_all();
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "Error posting to semaphore" } );
    }
    return 0;
}

void streamWriter::completeWrite( bool success )
{
    bool     stopped = false;
    uint64_t frameNo;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_writeMutex );
        frameNo = m_currSaveStopFrameNo;
        if( !success )
        {
            m_writerFailed = true;
            m_shutdown     = 1;
            m_writing      = NOT_WRITING;
        }
        else if( m_writing == STOP_WRITING && ( m_stopQueued || m_chunkFrames == 0 ) )
        {
            m_writing           = m_resumeAfterReconnect ? START_WRITING : NOT_WRITING;
            stopped             = !m_resumeAfterReconnect;
            m_stopWriteDeadline = 0;
        }
        m_stopQueued   = false;
        m_writePending = false;
    }
    if( stopped )
    {
        log<saving_stop>( { 0, frameNo } );
    }
    recordSavingState();
    m_writeCond.notify_all();
}

void streamWriter::fgThreadStart( streamWriter *o )
{
    o->fgThreadExec();
}

void streamWriter::fgThreadExec()
{
    m_fgStopped = false;
    fgThreadRun();
    m_fgReady   = false;
    m_fgStopped = true;
    m_writeCond.notify_all();
}

void streamWriter::fgThreadRun()
{
    m_fgThreadID = syscall( SYS_gettid );

    // Wait fpr the thread starter to finish initializing this thread.
    while( m_fgThreadInit == true && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    timespec missing_ts;

    IMAGE image;
    ino_t inode = 0; // The inode of the image stream file

    bool opened = false;

    while( m_shutdown == 0 )
    {
        /* Initialize ImageStreamIO
         */
        m_fgReady = false;
        opened    = false;
        m_restart = false; // Set this up front, since we're about to restart.

        sem_t *sem{ nullptr }; ///< The semaphore to monitor for new image data

        int logged = 0;
        while( !opened && !m_shutdown && !m_restart )
        {
            // b/c ImageStreamIO prints every single time, and latest version don't support stopping it yet,
            // and that isn't thread-safe-able anyway
            // we do our own checks.  This is the same code in ImageStreamIO_openIm...
            int  SM_fd;
            char SM_fname[200];
            ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), m_shmimName.c_str() );
            SM_fd = open( SM_fname, O_RDWR );
            if( SM_fd == -1 )
            {
                if( !logged )
                {
                    log<text_log>( "ImageStream " + m_shmimName + " not found (yet).  Retrying . . .",
                                   logPrio::LOG_NOTICE );
                }
                logged = 1;
                sleep( 1 ); // be patient
                continue;
            }

            // Found and opened,  close it and then use ImageStreamIO
            logged = 0;
            close( SM_fd );

            if( ImageStreamIO_openIm( &image, m_shmimName.c_str() ) == 0 )
            {
                if( image.md[0].sem < SEMAPHORE_MAXVAL )
                {
                    ImageStreamIO_closeIm( &image );
                    mx::sys::sleep( 1 ); // We just need to wait for the server process to finish startup.
                }
                else
                {
                    opened = true;

                    char SM_fname[200];
                    ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), m_shmimName.c_str() );

                    struct stat buffer;
                    int         rv = stat( SM_fname, &buffer );

                    if( rv != 0 )
                    {
                        log<software_critical>( { __FILE__,
                                                  __LINE__,
                                                  errno,
                                                  "Could not get inode for " + m_shmimName +
                                                      ". Source process will need to be restarted." } );
                        ImageStreamIO_closeIm( &image );
                        return;
                    }
                    inode = buffer.st_ino;
                }
            }
            else
            {
                mx::sys::sleep( 1 ); // be patient
            }
        }

        if( m_restart )
            continue; // this is kinda dumb.  we just go around on restart, so why test in the while loop at all?

        if( m_shutdown || !opened )
        {
            if( !opened )
                return;

            ImageStreamIO_closeIm( &image );
            return;
        }

        // now get a good semaphore
        m_semaphoreNumber =
            ImageStreamIO_getsemwaitindex( &image, m_semaphoreNumber ); // ask for semaphore we had before

        if( m_semaphoreNumber < 0 )
        {
            log<software_critical>(
                { __FILE__,
                  __LINE__,
                  "No valid semaphore found for " + m_shmimName + ". Source process will need to be restarted." } );
            return;
        }

        log<software_info>( { __FILE__,
                              __LINE__,
                              "got semaphore index " + std::to_string( m_semaphoreNumber ) + " for " + m_shmimName } );

        ImageStreamIO_semflush( &image, m_semaphoreNumber );

        sem = image.semptr[m_semaphoreNumber]; ///< The semaphore to monitor for new image data

        m_dataType = image.md[0].datatype;
        m_typeSize = ImageStreamIO_typesize( m_dataType );
        m_width    = image.md[0].size[0];
        m_height   = image.md[0].size[1];
        size_t length;
        if( image.md[0].naxis == 3 )
        {
            length = image.md[0].size[2];
        }
        else
        {
            length = 1;
        }
        std::cerr << "connected" << " " << m_width << "x" << m_height << "x" << (int)m_dataType << " (" << m_typeSize
                  << ")" << std::endl;

        // Now allocate the circBuffs
        if( allocate_circbufs() < 0 )
        {
            return; // will cause shutdown!
        }

        // And allocate the xrifs
        if( allocate_xrif() < 0 )
        {
            return; // Will cause shutdown!
        }

        // Buffer and XRIF setup can take long enough for monitor streams to accumulate
        // stale semaphore posts. Flush again and re-baseline counters so we do not
        // mistake startup backlog for repeated identical frames.
        ImageStreamIO_semflush( &image, m_semaphoreNumber );

        uint8_t  atype;
        size_t   snx, sny, snz;
        bool     useFrameArrays = image.cntarray != nullptr && length > 1;
        bool     useCnt1        = length > 1;
        bool     streamIsCube   = image.md[0].naxis == 3;
        size_t   frameBytes     = m_width * m_height * m_typeSize;
        uint64_t maxChunkTimeNs = static_cast<uint64_t>( m_maxChunkTime * 1e9 );

        uint64_t curr_image; // The current cnt1 index
        {                    // mutex scope
            std::lock_guard<std::mutex> lock( m_writeMutex );
            m_currImage      = 0;
            m_currChunkStart = 0;
            m_nextChunkStart = 0;
            m_chunkFrames    = 0;
            m_stopQueued     = false;
        }

        // Initialize curr_image after the post-setup flush.
        if( useCnt1 )
        {
            curr_image = image.md[0].cnt1;
        }
        else
        {
            curr_image = 0;
        }

        uint64_t last_cnt0; // = ((uint64_t)-1);

        // so we can initialize last_cnt0 to avoid frame skip on startup
        if( useFrameArrays )
        {
            last_cnt0 = image.cntarray[curr_image];
        }
        else
        {
            last_cnt0 = image.md[0].cnt0;
        }

        int cnt0flag = 0;

        bool restartWriting = false; // flag to prevent logging on a logging restart

        m_fgReady = true;

        // This is the main image grabbing loop.
        while( !m_shutdown && !m_restart )
        {
            timespec ts;
            XWC_SEM_WAIT_TS_RETVOID( ts, m_semWaitSec, m_semWaitNSec );

            if( sem_timedwait( sem, &ts ) == 0 )
            {
                // Drain
                while( sem_trywait( sem ) == 0 )
                {
                }

                if( errno != EAGAIN && errno != EINTR )
                {
                    if( !m_shutdown && !m_restart )
                    {
                        log<software_error>( { __FILE__, __LINE__, errno, "sem_trywait" } );
                    }
                    break;
                }

                std::unique_lock<std::mutex> lock( m_writeMutex );
                if( m_writePending && m_currImage >= m_currSaveStart && m_currImage < m_currSaveStop )
                {
                    const uint64_t previousFrame = m_currSaveStopFrameNo;
                    lock.unlock();
                    const bool completed = waitForWriteCompletion( previousFrame );
                    lock.lock();
                    if( !completed )
                    {
                        m_shutdown = 1;
                        return;
                    }
                }

                if( useCnt1 )
                {
                    curr_image = image.md[0].cnt1;
                }
                else
                {
                    curr_image = 0;
                }

                atype = image.md[0].datatype;
                snx   = image.md[0].size[0];
                sny   = image.md[0].size[1];
                if( streamIsCube )
                {
                    snz = image.md[0].size[2];
                }
                else
                {
                    snz = 1;
                }

                if( atype != m_dataType || snx != m_width || sny != m_height || snz != length )
                {
                    break; // exit the nearest while loop and get the new image setup.
                }

                if( m_shutdown || m_restart )
                {
                    break; // Check for exit signals
                }

                uint64_t new_cnt0;
                if( useFrameArrays )
                {
                    new_cnt0 = image.cntarray[curr_image];
                }
                else
                {
                    new_cnt0 = image.md[0].cnt0;
                }

                // clang-format off
                #ifdef SW_DEBUG
                std::cerr << "new_cnt0: " << new_cnt0 << "\n";
                #endif
                // clang-format on

                if( new_cnt0 == last_cnt0 )
                {
                    ++m_repeatSemaphoreCount;
                    ++cnt0flag;
                    if( cnt0flag > 10 ) // Because of the drain this shouldn't happen
                    {
                        m_restart = true; // if we get here 10 times then something else is wrong.
                    }
                    continue;
                }

                if( new_cnt0 - last_cnt0 > 1 ) //<- this is what we want to check.
                {
                    m_skippedFrameCount += ( new_cnt0 - last_cnt0 - 1 );
                }

                cnt0flag = 0;

                last_cnt0 = new_cnt0;

                char *curr_dest = m_rawImageCircBuff + m_currImage * frameBytes;
                char *curr_src  = reinterpret_cast<char *>( image.array.raw ) + curr_image * frameBytes;

                memcpy( curr_dest, curr_src, frameBytes );

                uint64_t *curr_timing = m_timingCircBuff + 5 * m_currImage;

                if( useFrameArrays )
                {
                    curr_timing[0] = image.cntarray[curr_image];
                    curr_timing[1] = image.atimearray[curr_image].tv_sec;
                    curr_timing[2] = image.atimearray[curr_image].tv_nsec;
                    curr_timing[3] = image.writetimearray[curr_image].tv_sec;
                    curr_timing[4] = image.writetimearray[curr_image].tv_nsec;
                }
                else
                {
                    curr_timing[0] = image.md[0].cnt0;
                    curr_timing[1] = image.md[0].atime.tv_sec;
                    curr_timing[2] = image.md[0].atime.tv_nsec;
                    curr_timing[3] = image.md[0].writetime.tv_sec;
                    curr_timing[4] = image.md[0].writetime.tv_nsec;
                }

                // Check if we need to time-stamp ourselves -- for old cacao streams
                if( curr_timing[1] == 0 )
                {

                    if( clock_gettime( CLOCK_REALTIME, &missing_ts ) < 0 )
                    {
                        log<software_critical>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
                        return;
                    }

                    curr_timing[1] = missing_ts.tv_sec;
                    curr_timing[2] = missing_ts.tv_nsec;
                }

                // just set w-time to a-time if it's missing
                if( curr_timing[3] == 0 )
                {
                    curr_timing[3] = curr_timing[1];
                    curr_timing[4] = curr_timing[2];
                }

                m_currImageTime = curr_timing[3] * 1000000000ULL + curr_timing[4];

                if( m_shutdown && m_writing == WRITING )
                {
                    m_writing           = STOP_WRITING;
                    m_stopWriteDeadline = 0;
                }

                if( m_writing == START_WRITING )
                {
                    if( !restartWriting && !m_resumeAfterReconnect )
                    {
                        log<saving_start>( { 1, new_cnt0 } );
                    }
                    restartWriting         = false;
                    m_resumeAfterReconnect = false;
                    m_writing              = WRITING;
                }

                if( m_writing == WRITING || ( m_writing == STOP_WRITING && !m_stopQueued ) )
                {
                    if( m_chunkFrames == 0 )
                    {
                        m_currChunkStart     = m_currImage;
                        m_nextChunkStart     = ( m_currImage / m_writeChunkLength ) * m_writeChunkLength;
                        m_currChunkStartTime = m_currImageTime;
                    }
                    ++m_chunkFrames;
                    const bool full    = m_currImage - m_nextChunkStart == m_writeChunkLength - 1;
                    const bool expired = m_currImageTime >= m_currChunkStartTime &&
                                         m_currImageTime - m_currChunkStartTime > maxChunkTimeNs;
                    if( full || expired || m_writing == STOP_WRITING )
                    {
                        if( queueWrite( lock, new_cnt0 ) < 0 )
                        {
                            return;
                        }
                        if( !full && expired && m_writing == WRITING )
                        {
                            m_writing      = START_WRITING;
                            restartWriting = true;
                        }
                    }
                }

                ++m_currImage;
                if( m_currImage >= m_circBuffLength )
                {
                    m_currImage = 0;
                }
            }
            else
            {
                const int waitError = errno;
                { //mutex scope
                    std::unique_lock<std::mutex> lock( m_writeMutex );
                    const double                 now = mx::sys::get_curr_time();
                    const bool                   expired =
                        m_writing == WRITING && m_chunkFrames > 0 && now * 1e9 - m_currChunkStartTime > maxChunkTimeNs;
                    const bool stopping = m_writing == STOP_WRITING && !m_stopQueued && now >= m_stopWriteDeadline;
                    if( ( expired || stopping ) && m_chunkFrames > 0 )
                    {
                        if( queueWrite( lock, last_cnt0 ) < 0 )
                        {
                            return;
                        }
                        if( m_writing == WRITING )
                        {
                            m_writing      = START_WRITING;
                            restartWriting = true;
                        }
                    }
                }
                errno = waitError;

                if( image.md[0].sem <= 0 )
                {
                    break; // Indicates that the server has cleaned up.
                }

                // Check for why we timed out
                if( errno == EINTR )
                {
                    break; // This will indicate time to shutdown, loop will exit normally flags set.
                }

                // ETIMEDOUT just means we should wait more.
                // Otherwise, report an error.
                if( errno != ETIMEDOUT )
                {
                    log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
                    break;
                }

                // Check if the file has disappeared.
                int  SM_fd;
                char SM_fname[200];
                ImageStreamIO_filename( SM_fname, sizeof( SM_fname ), m_shmimName.c_str() );
                SM_fd = open( SM_fname, O_RDWR );
                if( SM_fd == -1 )
                {
                    m_restart = true;
                }
                close( SM_fd );

                // Check if the inode changed
                struct stat buffer;
                int         rv = stat( SM_fname, &buffer );
                if( rv != 0 )
                {
                    m_restart = true;
                }

                else if( buffer.st_ino != inode )
                {
#ifdef SW_DEBUG
                    std::cerr << "Restarting due to inode . . . \n";
#endif
                    m_restart = true;
                }
            }
        }

        m_fgReady = false;
        { //mutex scope
            std::unique_lock<std::mutex> lock( m_writeMutex );
            const bool                   resume = !m_shutdown && ( m_writing == WRITING || m_writing == START_WRITING );
            m_resumeAfterReconnect              = resume;
            if( m_chunkFrames > 0 )
            {
                m_writing           = STOP_WRITING;
                m_stopWriteDeadline = 0;
                if( queueWrite( lock, last_cnt0 ) < 0 )
                {
                    return;
                }
            }
            else if( m_writePending )
            {
                m_writing    = STOP_WRITING;
                m_stopQueued = true;
            }
            else
            {
                m_writing = resume ? START_WRITING : NOT_WRITING;
            }
        }
        if( !waitForWriteCompletion( last_cnt0 ) )
        {
            m_shutdown = 1;
            return;
        }

        release_circbufs();

        if( opened )
        {
            if( m_semaphoreNumber >= 0 )
            {
                ///\todo is this release necessary with closeIM?
                image.semReadPID[m_semaphoreNumber] = 0; // release semaphore
            }
            ImageStreamIO_closeIm( &image );
            opened = false;
        }

    } // outer loop, will exit if m_shutdown==true

    // One more check
    if( m_rawImageCircBuff )
    {
        free( m_rawImageCircBuff );
        m_rawImageCircBuff = 0;
    }

    if( m_timingCircBuff )
    {
        free( m_timingCircBuff );
        m_timingCircBuff = 0;
    }
}

void streamWriter::swThreadStart( streamWriter *s )
{
    s->swThreadExec();
}

void streamWriter::swThreadExec()
{
    m_swThreadID = syscall( SYS_gettid );

    // Wait fpr the thread starter to finish initializing this thread.
    while( m_swThreadInit == true && m_shutdown == 0 )
    {
        sleep( 1 );
    }

    while( !m_shutdown || !m_fgStopped || m_writePending )
    {
        while( !shutdown() && ( !( state() == stateCodes::READY || state() == stateCodes::OPERATING ) ) )
        {
            sleep( 1 );
        }

        if( shutdown() && m_fgStopped && !m_writePending )
        {
            break;
        }

        timespec ts;

        if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
        {
            log<software_critical>( { __FILE__, __LINE__, errno, 0, "clock_gettime" } );
            return; // will trigger a shutdown
        }

        ts.tv_sec += m_semWaitSec;
        mx::sys::timespecAddNsec( ts, m_semWaitNSec );

        if( sem_timedwait( &m_swSemaphore, &ts ) == 0 )
        {
            if( m_writePending && doEncode() < 0 )
            {
                log<software_critical>( { __FILE__, __LINE__, "error encoding data" } );
                return;
            }
            // Otherwise, success, and we just go on.
        }
        else
        {
            // Check for why we timed out
            if( errno == EINTR )
            {
                continue; // This will probably indicate time to shutdown, loop will exit normally if flags set.
            }

            // ETIMEDOUT just means we should wait more.
            // Otherwise, report an error.
            if( errno != ETIMEDOUT )
            {
                log<software_error>( { __FILE__, __LINE__, errno, "sem_timedwait" } );
                break;
            }
        }
    } // outer loop, will exit if m_shutdown==true
}

int streamWriter::doEncode()
{
    uint64_t saveStart, saveStop;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_writeMutex );
        if( m_writing == NOT_WRITING )
        {
            m_writePending = false;
            m_writeCond.notify_all();
            return 0;
        }
        saveStart      = m_currSaveStart;
        saveStop       = m_currSaveStop;
        m_writePending = true;
    }

    recordSavingState( true );
    if( saveStart > saveStop || saveStop > m_circBuffLength || saveStop - saveStart > m_writeChunkLength )
    {
        completeWrite( false );
        return log<software_critical, -1>( { __FILE__, __LINE__, "invalid save window" } );
    }
    const size_t nFrames = saveStop - saveStart;
    if( nFrames == 0 )
    {
        if( m_xrif )
        {
            recordSavingStats( true );
        }
        completeWrite();
        return 0;
    }

    constexpr size_t timingBytes = 5 * sizeof( uint64_t );
    const size_t     maxSize     = std::numeric_limits<size_t>::max();
    if( m_width == 0 || m_height == 0 || m_typeSize <= 0 || m_width > maxSize / m_height ||
        m_width * m_height > maxSize / static_cast<size_t>( m_typeSize ) )
    {
        completeWrite( false );
        return log<software_critical, -1>( { __FILE__, __LINE__, "invalid frame byte count" } );
    }
    const size_t nBytes = m_width * m_height * m_typeSize;
    if( m_circBuffLength > maxSize / nBytes || m_circBuffLength > maxSize / timingBytes ||
        nFrames > std::numeric_limits<xrif_dimension_t>::max() || !m_rawImageCircBuff || !m_timingCircBuff || !m_xrif ||
        !m_xrif_timing || !m_xrif->raw_buffer || !m_xrif_timing->raw_buffer ||
        nFrames > m_xrif->raw_buffer_size / nBytes || nFrames > m_xrif_timing->raw_buffer_size / timingBytes )
    {
        completeWrite( false );
        return log<software_critical, -1>( { __FILE__, __LINE__, "save window exceeds allocated buffer capacity" } );
    }

    // Both sizing operations must succeed before either destination is modified.
    int rv = xrif_set_size( m_xrif, m_width, m_height, 1, nFrames, m_dataType );
    if( rv != XRIF_NOERROR )
    {
        completeWrite( false );
        return log<software_alert, -1>( { __FILE__, __LINE__, 0, rv, "xrif set size error. DATA POSSIBLY LOST" } );
    }
    rv = xrif_set_size( m_xrif_timing, 5, 1, 1, nFrames, XRIF_TYPECODE_UINT64 );
    if( rv != XRIF_NOERROR )
    {
        completeWrite( false );
        return log<software_alert, -1>( { __FILE__, __LINE__, 0, rv, "xrif set size error. DATA POSSIBLY LOST" } );
    }

    rv = xrif_set_lz4_acceleration( m_xrif, m_lz4accel );
    if( rv != XRIF_NOERROR )
    {
        // This may just be out of range, it's only an error.
        log<software_error>( { __FILE__, __LINE__, 0, rv, "xrif set LZ4 acceleration error." } );
    }

    memcpy( m_xrif->raw_buffer, m_rawImageCircBuff + saveStart * nBytes, nFrames * nBytes );

    rv = xrif_set_lz4_acceleration( m_xrif_timing, m_lz4accel );
    if( rv != XRIF_NOERROR )
    {
        // This may just be out of range, it's only an error.
        log<software_error>( { __FILE__, __LINE__, 0, rv, "xrif set LZ4 acceleration error." } );
    }

#ifdef SW_DEBUG
    for( size_t nF = 0; nF < nFrames; ++nF )
    {
        std::cerr << "      " << ( m_timingCircBuff + saveStart * 5 + nF * 5 )[0] << "\n";
    }
#endif

    memcpy( m_xrif_timing->raw_buffer, m_timingCircBuff + saveStart * 5, nFrames * 5 * sizeof( uint64_t ) );

    rv = xrif_encode( m_xrif );
    if( rv != XRIF_NOERROR )
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>( { __FILE__, __LINE__, 0, rv, "xrif encode error. DATA POSSIBLY LOST." } );
    }

    rv = xrif_write_header( m_xrif_header, m_xrif );
    if( rv != XRIF_NOERROR )
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>( { __FILE__, __LINE__, 0, rv, "xrif write header error. DATA POSSIBLY LOST." } );
    }

    rv = xrif_encode( m_xrif_timing );
    if( rv != XRIF_NOERROR )
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>( { __FILE__, __LINE__, 0, rv, "xrif encode error. DATA POSSIBLY LOST." } );
    }

    rv = xrif_write_header( m_xrif_timing_header, m_xrif_timing );
    if( rv != XRIF_NOERROR )
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>( { __FILE__, __LINE__, 0, rv, "xrif write header error. DATA POSSIBLY LOST" } );
    }

    // Now break down the acq time of the first image in the buffer for use in file name
    // tm        uttime; // The broken down time.
    timespec *fts = reinterpret_cast<timespec *>( m_timingCircBuff + saveStart * 5 + 1 );

    std::string fileName;
    std::string relPath;
    mx::error_t errc = file::fileTimeRelPath( fileName, relPath, m_outName, "xrif", fts->tv_sec, fts->tv_nsec );
    if( errc != mx::error_t::noerror )
    {
        std::string msg = "error from file::fileTimeRePath: ";
        msg += mx::errorMessage( errc );
        msg += " (" + std::string( mx::errorName( errc ) ) + ")";
        completeWrite( false );
        return log<software_error, -1>( { __FILE__, __LINE__, msg } );
    }

    std::string fullPath = m_rawimageDir + '/' + relPath;

    try
    {
        std::filesystem::create_directories( fullPath ); // this does nothing if fname already exists
    }
    catch( const std::filesystem::filesystem_error &e )
    {
        std::string msg = "filesystem_error from std::create_directories. ";
        msg += e.what();
        msg += " code: ";
        msg += e.code().value();
        completeWrite( false );
        return log<software_critical, -1>( { __FILE__, __LINE__, msg } );
    }
    catch( const std::exception &e )
    {
        std::string msg = "exception from std::create_directories. ";
        msg += e.what();
        completeWrite( false );
        return log<software_critical, -1>( { __FILE__, __LINE__, msg } );
    }

    m_outFilePath = fullPath + '/' + fileName;
    FILE *fp_xrif = fopen( m_outFilePath.c_str(), "wb" );
    if( fp_xrif == NULL )
    {
        // This is it.  If we can't write data to disk need to fix.
        completeWrite( false );
        return log<software_critical, -1>( { __FILE__, __LINE__, errno, 0, "failed to open file for writing" } );
    }

    size_t bw = fwrite( m_xrif_header, sizeof( uint8_t ), XRIF_HEADER_SIZE, fp_xrif );

    if( bw != XRIF_HEADER_SIZE )
    {
        log<software_alert>( { __FILE__,
                               __LINE__,
                               errno,
                               0,
                               "failure writing header to file.  DATA LOSS LIKELY. bytes = " + std::to_string( bw ) } );
        // We go on . . .
    }

    bw = fwrite( m_xrif->raw_buffer, sizeof( uint8_t ), m_xrif->compressed_size, fp_xrif );

    if( bw != m_xrif->compressed_size )
    {
        log<software_alert>( { __FILE__,
                               __LINE__,
                               errno,
                               0,
                               "failure writing data to file.  DATA LOSS LIKELY. bytes = " + std::to_string( bw ) } );
    }

    bw = fwrite( m_xrif_timing_header, sizeof( uint8_t ), XRIF_HEADER_SIZE, fp_xrif );

    if( bw != XRIF_HEADER_SIZE )
    {
        log<software_alert>(
            { __FILE__,
              __LINE__,
              errno,
              0,
              "failure writing timing header to file.  DATA LOSS LIKELY.  bytes = " + std::to_string( bw ) } );
    }

    bw = fwrite( m_xrif_timing->raw_buffer, sizeof( uint8_t ), m_xrif_timing->compressed_size, fp_xrif );

    if( bw != m_xrif_timing->compressed_size )
    {
        log<software_alert>(
            { __FILE__,
              __LINE__,
              errno,
              0,
              "failure writing timing data to file. DATA LOSS LIKELY. bytes = " + std::to_string( bw ) } );
    }

    fclose( fp_xrif );

    recordSavingStats( true );

    completeWrite();
    return 0;

} // doEncode

INDI_NEWCALLBACK_DEFN( streamWriter, m_indiP_writing )
( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_writing, ipRecv );

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    bool     stopped   = false;
    uint64_t stopFrame = 0;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_writeMutex );
        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off && m_writing == STOP_WRITING )
        {
            m_resumeAfterReconnect = false;
        }
        if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off &&
            ( m_writing == WRITING || m_writing == START_WRITING ) )
        {
            m_resumeAfterReconnect = false;
            m_stopWriteDeadline    = mx::sys::get_curr_time() + m_writeStopTimeout;
            if( m_chunkFrames == 0 )
            {
                m_writing           = m_writePending ? STOP_WRITING : NOT_WRITING;
                m_stopQueued        = m_writePending;
                m_stopWriteDeadline = 0;
                stopped             = !m_writePending;
                stopFrame           = m_currSaveStopFrameNo;
            }
            else
            {
                m_writing = STOP_WRITING;
            }
        }
        else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On && m_writing == NOT_WRITING )
        {
            m_writing              = START_WRITING;
            m_chunkFrames          = 0;
            m_stopQueued           = false;
            m_resumeAfterReconnect = false;
            m_stopWriteDeadline    = 0;
        }
    }

    if( stopped )
    {
        log<saving_stop>( { 0, stopFrame } );
        recordSavingState();
    }

    return 0;
}

void streamWriter::updateINDI()
{
    int    writing;
    double ratio, encodeRate, differenceRate, reorderRate, compressRate;
    size_t frameBytes;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_writeMutex );
        writing        = m_writing;
        ratio          = m_compressionRatio;
        encodeRate     = m_encodeRate;
        differenceRate = m_differenceRate;
        reorderRate    = m_reorderRate;
        compressRate   = m_compressRate;
        frameBytes     = m_frameBytes;
    }
    // Only update this if not changing
    if( writing == NOT_WRITING || writing == WRITING || writing == START_WRITING )
    {
        if( writing == WRITING || writing == START_WRITING )
        {
            indi::updateSwitchIfChanged( m_indiP_writing, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK );
            indi::updateIfChanged( m_indiP_xrifStats, "ratio", ratio, m_indiDriver, INDI_BUSY );
            indi::updateIfChanged( m_indiP_xrifStats, "encodeMBsec", encodeRate / 1048576.0, m_indiDriver, INDI_BUSY );
            indi::updateIfChanged( m_indiP_xrifStats,
                                   "encodeFPS",
                                   encodeRate / std::max<size_t>( frameBytes, 1 ),
                                   m_indiDriver,
                                   INDI_BUSY );
            indi::updateIfChanged(
                m_indiP_xrifStats, "differenceMBsec", differenceRate / 1048576.0, m_indiDriver, INDI_BUSY );
            indi::updateIfChanged( m_indiP_xrifStats,
                                   "differenceFPS",
                                   differenceRate / std::max<size_t>( frameBytes, 1 ),
                                   m_indiDriver,
                                   INDI_BUSY );
            indi::updateIfChanged(
                m_indiP_xrifStats, "reorderMBsec", reorderRate / 1048576.0, m_indiDriver, INDI_BUSY );
            indi::updateIfChanged( m_indiP_xrifStats,
                                   "reorderFPS",
                                   reorderRate / std::max<size_t>( frameBytes, 1 ),
                                   m_indiDriver,
                                   INDI_BUSY );
            indi::updateIfChanged(
                m_indiP_xrifStats, "compressMBsec", compressRate / 1048576.0, m_indiDriver, INDI_BUSY );
            indi::updateIfChanged( m_indiP_xrifStats,
                                   "compressFPS",
                                   compressRate / std::max<size_t>( frameBytes, 1 ),
                                   m_indiDriver,
                                   INDI_BUSY );
        }
        else
        {
            indi::updateSwitchIfChanged( m_indiP_writing, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_OK );
            indi::updateIfChanged( m_indiP_xrifStats, "ratio", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "encodeMBsec", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "encodeFPS", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "differenceMBsec", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "differenceFPS", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "reorderMBsec", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "reorderFPS", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "compressMBsec", 0.0, m_indiDriver, INDI_IDLE );
            indi::updateIfChanged( m_indiP_xrifStats, "compressFPS", 0.0, m_indiDriver, INDI_IDLE );
        }
    }
}

int streamWriter::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_saving_state() );
}

int streamWriter::recordTelem( const telem_saving_state * )
{
    return recordSavingState( true );
}

int streamWriter::recordSavingState( bool force )
{
    std::lock_guard<std::mutex> lock( m_writeMutex );

    uint64_t frameNo = m_currSaveStopFrameNo;

    int16_t state;
    if( m_writing == WRITING || m_writing == START_WRITING ||
        m_writing == STOP_WRITING ) // Changed from just writing 5/2024
        state = 1;
    else
        state = 0;

    if( state != m_lastSavingState || frameNo != m_lastSavingFrame || force )
    {
        telem<telem_saving_state>( { state, frameNo } );

        m_lastSavingState = state;
        m_lastSavingFrame = frameNo;
    }

    return 0;
}

int streamWriter::recordSavingStats( bool force )
{
    std::lock_guard<std::mutex> lock( m_writeMutex );
    m_compressionRatio = m_xrif->compression_ratio;
    m_encodeRate       = m_xrif->encode_rate;
    m_differenceRate   = m_xrif->difference_rate;
    m_reorderRate      = m_xrif->reorder_rate;
    m_compressRate     = m_xrif->compress_rate;
    m_frameBytes       = m_width * m_height * m_typeSize;

    static uint32_t last_rawSize        = -1;
    static uint32_t last_compressedSize = -1;
    static float    last_encodeRate     = -1;
    static float    last_differenceRate = -1;
    static float    last_reorderRate    = -1;
    static float    last_compressRate   = -1;

    if( m_xrif->raw_size != last_rawSize || m_xrif->compressed_size != last_compressedSize ||
        m_xrif->encode_rate != last_encodeRate || m_xrif->difference_rate != last_differenceRate ||
        m_xrif->reorder_rate != last_reorderRate || m_xrif->compress_rate != last_compressRate || force )
    {
        telem<telem_saving>( { (uint32_t)m_xrif->raw_size,
                               (uint32_t)m_xrif->compressed_size,
                               (float)m_xrif->encode_rate,
                               (float)m_xrif->difference_rate,
                               (float)m_xrif->reorder_rate,
                               (float)m_xrif->compress_rate } );

        last_rawSize        = m_xrif->raw_size;
        last_compressedSize = m_xrif->compressed_size;
        last_encodeRate     = m_xrif->encode_rate;
        last_differenceRate = m_xrif->difference_rate;
        last_reorderRate    = m_xrif->reorder_rate;
        last_compressRate   = m_xrif->compress_rate;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif
