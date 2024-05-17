/** \file streamWriter.hpp
 * \brief The MagAO-X Image Stream Writer
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 *
 * \ingroup streamWriter_files
 */

#ifndef streamWriter_hpp
#define streamWriter_hpp

#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <xrif/xrif.h>

#include <mx/sys/timeUtils.hpp>

#include "../../libMagAOX/libMagAOX.hpp" //Note this is included on command line to trigger pch
#include "../../magaox_git_version.h"

#define NOT_WRITING (0)
#define START_WRITING (1)
#define WRITING (2)
#define STOP_WRITING (3)

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

protected:
    /** \name configurable parameters
     *@{
     */

    std::string m_rawimageDir; ///< The path where files will be saved.

    size_t m_circBuffLength{1024}; ///< The length of the circular buffer, in frames

    size_t m_writeChunkLength{512}; ///< The number of frames to write at a time

    double m_maxChunkTime{10}; ///< The maximum time before writing regardless of number of frames.

    std::string m_shmimName; ///< The name of the shared memory buffer.

    std::string m_outName; ///< The name to use for outputting files,  Default is m_shmimName.

    int m_semaphoreNumber{7}; ///< The image structure semaphore index.

    unsigned m_semWaitSec{0}; ///< The time in whole sec to wait on the semaphore, to which m_semWaitNSec is added.  Default is 0 nsec.

    unsigned m_semWaitNSec{500000000}; ///< The time in nsec to wait on the semaphore, added to m_semWaitSec.  Max is 999999999. Default is 5e8 nsec.

    int m_lz4accel{1};

    bool m_compress{true};

    ///@}

    size_t m_width{0};     ///< The width of the image
    size_t m_height{0};    ///< The height of the image
    uint8_t m_dataType{0}; ///< The ImageStreamIO type code.
    int m_typeSize{0};     ///< The pixel byte depth

    char *m_rawImageCircBuff{nullptr};
    uint64_t *m_timingCircBuff{nullptr};

    size_t m_currImage{0};

    double m_currImageTime{0}; ///< The write-time of the current image

    double m_currChunkStartTime{0}; ///< The write-time of the first image in the chunk

    // Writer book-keeping:
    int m_writing{NOT_WRITING}; ///< Controls whether or not images are being written, and sequences start and stop of writing.

    uint64_t m_currChunkStart{0}; ///< The circular buffer starting position of the current to-be-written chunk.
    uint64_t m_nextChunkStart{0}; ///< The circular buffer starting position of the next to-be-written chunk.

    uint64_t m_currSaveStart{0}; ///< The circular buffer position at which to start saving.
    uint64_t m_currSaveStop{0};  ///< The circular buffer position at which to stop saving.

    uint64_t m_currSaveStopFrameNo{0};  ///< The frame number of the image at which saving stopped (for logging)

    /// The xrif compression handle for image data
    xrif_t m_xrif{nullptr};

    /// Storage for the xrif image data file header
    char *m_xrif_header{nullptr};

    /// The xrif compression handle for image data
    xrif_t m_xrif_timing{nullptr};

    /// Storage for the xrif image data file header
    char *m_xrif_timing_header{nullptr};

public:
    /// Default c'tor
    streamWriter();

    /// Destructor
    ~streamWriter() noexcept;

    /// Setup the configuration system (called by MagAOXApp::setup())
    virtual void setupConfig();

    /// load the configuration system results (called by MagAOXApp::setup())
    virtual void loadConfig();

    /// Startup functions
    /** Sets up the INDI vars.
     *
     */
    virtual int appStartup();

    /// Implementation of the FSM for the Siglent SDG
    virtual int appLogic();

    /// Do any needed shutdown tasks.  Currently nothing in this app.
    virtual int appShutdown();

protected:
    /** \name SIGSEGV & SIGBUS signal handling
     * These signals occur as a result of a ImageStreamIO source server resetting (e.g. changing frame sizes).
     * When they occur a restart of the framegrabber and framewriter thread main loops is triggered.
     *
     * @{
     */
    bool m_restart{false};

    static streamWriter *m_selfWriter; ///< Static pointer to this (set in constructor).  Used for getting out of the static SIGSEGV handler.

    /// Initialize the xrif system.
    /** Allocates the handles and headers pointers.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int initialize_xrif();

    /// Sets the handler for SIGSEGV and SIGBUS
    /** These are caused by ImageStreamIO server resets.
     */
    int setSigSegvHandler();

    /// The handler called when SIGSEGV or SIGBUS is received, which will be due to ImageStreamIO server resets.  Just a wrapper for handlerSigSegv.
    static void _handlerSigSegv(int signum,
                                siginfo_t *siginf,
                                void *ucont);

    /// Handles SIGSEGV and SIGBUS.  Sets m_restart to true.
    void handlerSigSegv(int signum,
                        siginfo_t *siginf,
                        void *ucont);
    ///@}

    /** \name Framegrabber Thread
     * This thread monitors the ImageStreamIO buffer and copies its images to the circular buffer.
     *
     * @{
     */
    int m_fgThreadPrio{1}; ///< Priority of the framegrabber thread, should normally be > 00.

    std::string m_fgCpuset; ///< The cpuset for the framegrabber thread.  Ignored if empty (the default).

    std::thread m_fgThread; ///< A separate thread for the actual framegrabbings

    bool m_fgThreadInit{true}; ///< Synchronizer to ensure f.g. thread initializes before doing dangerous things.

    pid_t m_fgThreadID{0}; ///< F.g. thread PID.

    pcf::IndiProperty m_fgThreadProp; ///< The property to hold the f.g. thread details.

    /// Worker function to allocate the circular buffers.
    /** This takes place in the fg thread after connecting to the stream.
     *
     * \returns 0 on sucess.
     * \returns -1 on error.
     */
    int allocate_circbufs();

    /// Worker function to configure and allocate the xrif handles.
    /** This takes place in the fg thread after connecting to the stream.
     *
     * \returns 0 on sucess.
     * \returns -1 on error.
     */
    int allocate_xrif();

    /// Thread starter, called by fgThreadStart on thread construction.  Calls fgThreadExec.
    static void fgThreadStart(streamWriter *s /**< [in] a pointer to an streamWriter instance (normally this) */);

    /// Execute the frame grabber main loop.
    void fgThreadExec();

    ///@}

    /** \name Stream Writer Thread
     * This thread writes chunks of the circular buffer to disk.
     *
     * @{
     */
    int m_swThreadPrio{1}; ///< Priority of the stream writer thread, should normally be > 0, and <= m_fgThreadPrio.

    std::string m_swCpuset; ///< The cpuset for the framegrabber thread.  Ignored if empty (the default).

    sem_t m_swSemaphore; ///< Semaphore used to synchronize the fg thread and the sw thread.

    std::thread m_swThread; ///< A separate thread for the actual writing

    bool m_swThreadInit{true}; ///< Synchronizer to ensure s.w. thread initializes before doing dangerous things.

    pid_t m_swThreadID{0}; ///< S.w. thread pid.

    pcf::IndiProperty m_swThreadProp; ///< The property to hold the s.w. thread details.

    size_t m_fnameSz{0};

    char *m_fname{nullptr};

    std::string m_fnameBase;

    /// Thread starter, called by swThreadStart on thread construction.  Calls swThreadExec.
    static void swThreadStart(streamWriter *s /**< [in] a pointer to an streamWriter instance (normally this) */);

    /// Execute the stream writer main loop.
    void swThreadExec();

    /// Function called when semaphore is raised to do the encode and write.
    int doEncode();
    ///@}

    // INDI:
protected:
    // declare our properties
    pcf::IndiProperty m_indiP_writing;

    pcf::IndiProperty m_indiP_xrifStats;

public:
    INDI_NEWCALLBACK_DECL(streamWriter, m_indiP_writing);

    void updateINDI();

    /** \name Telemeter Interface
     *
     * @{
     */
    int checkRecordTimes();

    int recordTelem(const telem_saving_state *);

    int recordSavingState(bool force = false);
    int recordSavingStats(bool force = false);

    ///@}
};

// Set self pointer to null so app starts up uninitialized.
streamWriter *streamWriter::m_selfWriter = nullptr;

streamWriter::streamWriter() : MagAOXApp(MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED)
{
    m_powerMgtEnabled = false;

    m_selfWriter = this;

    return;
}

streamWriter::~streamWriter() noexcept
{
    if (m_xrif)
        xrif_delete(m_xrif);

    if (m_xrif_header)
        free(m_xrif_header);

    if (m_xrif_timing)
        xrif_delete(m_xrif_timing);

    if (m_xrif_timing_header)
        free(m_xrif_timing_header);

    return;
}

void streamWriter::setupConfig()
{
    config.add("writer.savePath", "", "writer.savePath", argType::Required, "writer", "savePath", false, "string", "The absolute path where images are saved. Will use MagAO-X default if not set.");

    config.add("writer.circBuffLength", "", "writer.circBuffLength", argType::Required, "writer", "circBuffLength", false, "size_t", "The length in frames of the circular buffer. Should be an integer multiple of and larger than writeChunkLength.");

    config.add("writer.writeChunkLength", "", "writer.writeChunkLength", argType::Required, "writer", "writeChunkLength", false, "size_t", "The length in frames of the chunks to write to disk. Should be smaller than circBuffLength.");

    config.add("writer.maxChunkTime", "", "writer.maxChunkTime", argType::Required, "writer", "maxChunkTime", false, "float", "The max length in seconds of the chunks to write to disk. Default is 60 sec.");

    config.add("writer.threadPrio", "", "writer.threadPrio", argType::Required, "writer", "threadPrio", false, "int", "The real-time priority of the stream writer thread.");

    config.add("writer.cpuset", "", "writer.cpuset", argType::Required, "writer", "cpuset", false, "int", "The cpuset for the writer thread.");

    config.add("writer.compress", "", "writer.compress", argType::Required, "writer", "compress", false, "bool", "Flag to set whether compression is used.  Default true.");

    config.add("writer.lz4accel", "", "writer.lz4accel", argType::Required, "writer", "lz4accel", false, "int", "The LZ4 acceleration parameter.  Larger is faster, but lower compression.");

    config.add("writer.outName", "", "writer.outName", argType::Required, "writer", "outName", false, "int", "The name to use for output files.  Default is the shmimName.");

    config.add("framegrabber.shmimName", "", "framegrabber.shmimName", argType::Required, "framegrabber", "shmimName", false, "int", "The name of the stream to monitor. From /tmp/shmimName.im.shm.");

    config.add("framegrabber.semaphoreNumber", "", "framegrabber.semaphoreNumber", argType::Required, "framegrabber", "semaphoreNumber", false, "int", "The semaphore to wait on. Default is 7.");

    config.add("framegrabber.semWait", "", "framegrabber.semWait", argType::Required, "framegrabber", "semWait", false, "int", "The time in nsec to wait on the semaphore.  Max is 999999999. Default is 5e8 nsec.");

    config.add("framegrabber.threadPrio", "", "framegrabber.threadPrio", argType::Required, "framegrabber", "threadPrio", false, "int", "The real-time priority of the framegrabber thread.");

    config.add("framegrabber.cpuset", "", "framegrabber.cpuset", argType::Required, "framegrabber", "cpuset", false, "string", "The cpuset for the framegrabber thread.");

    telemeterT::setupConfig(config);
}

void streamWriter::loadConfig()
{

    config(m_circBuffLength, "writer.circBuffLength");
    config(m_writeChunkLength, "writer.writeChunkLength");
    config(m_maxChunkTime, "writer.maxChunkTime");
    config(m_swThreadPrio, "writer.threadPrio");
    config(m_swCpuset, "writer.cpuset");
    config(m_compress, "writer.compress");
    config(m_lz4accel, "writer.lz4accel");
    if (m_lz4accel < XRIF_LZ4_ACCEL_MIN)
        m_lz4accel = XRIF_LZ4_ACCEL_MIN;
    if (m_lz4accel > XRIF_LZ4_ACCEL_MAX)
        m_lz4accel = XRIF_LZ4_ACCEL_MAX;

    config(m_shmimName, "framegrabber.shmimName");

    m_outName = m_shmimName;
    config(m_outName, "writer.outName");

    config(m_semaphoreNumber, "framegrabber.semaphoreNumber");
    config(m_semWaitNSec, "framegrabber.semWait");

    config(m_fgThreadPrio, "framegrabber.threadPrio");
    config(m_fgCpuset, "framegrabber.cpuset");

    // Set some defaults
    // Setup default log path
    m_rawimageDir = MagAOXPath + "/" + MAGAOX_rawimageRelPath + "/" + m_outName;
    config(m_rawimageDir, "writer.savePath");

    if (telemeterT::loadConfig(config) < 0)
    {
        log<text_log>("Error during telemeter config", logPrio::LOG_CRITICAL);
        m_shutdown = true;
    }
}

int streamWriter::appStartup()
{
    // Create save directory.
    errno = 0;
    if (mkdir(m_rawimageDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) < 0)
    {
        if (errno != EEXIST)
        {
            std::stringstream logss;
            logss << "Failed to create image directory (" << m_rawimageDir << ").  Errno says: " << strerror(errno);
            log<software_critical>({__FILE__, __LINE__, errno, 0, logss.str()});

            return -1;
        }
    }

    // set up the  INDI properties
    createStandardIndiToggleSw(m_indiP_writing, "writing");
    registerIndiPropertyNew(m_indiP_writing, INDI_NEWCALLBACK(m_indiP_writing));

    // Register the stats INDI property
    REG_INDI_NEWPROP_NOCB(m_indiP_xrifStats, "xrif", pcf::IndiProperty::Number);
    m_indiP_xrifStats.setLabel("xrif compression performance");

    indi::addNumberElement<float>(m_indiP_xrifStats, "ratio", 0, 1.0, 0.0, "%0.2f", "Compression Ratio");

    indi::addNumberElement<float>(m_indiP_xrifStats, "differenceMBsec", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Differencing Rate [MB/sec]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "reorderMBsec", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Reordering Rate [MB/sec]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "compressMBsec", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Compression Rate [MB/sec]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "encodeMBsec", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Total Encoding Rate [MB/sec]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "differenceFPS", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Differencing Rate [f.p.s.]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "reorderFPS", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Reordering Rate [f.p.s.]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "compressFPS", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Compression Rate [f.p.s.]");

    indi::addNumberElement<float>(m_indiP_xrifStats, "encodeFPS", 0, std::numeric_limits<float>::max(), 0.0, "%0.2f", "Total Encoding Rate [f.p.s.]");

    // Now set up the framegrabber and writer threads.
    //  - need SIGSEGV and SIGBUS handling for ImageStreamIO restarts
    //  - initialize the semaphore
    //  - start the threads

    if (setSigSegvHandler() < 0)
        return log<software_error, -1>({__FILE__, __LINE__});

    if (sem_init(&m_swSemaphore, 0, 0) < 0)
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "Initializing S.W. semaphore"});

    // Check if we have a safe writeChunkLengthh
    if (m_circBuffLength % m_writeChunkLength != 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, "Write chunk length is not a divisor of circular buffer length."});
    }

    if (initialize_xrif() < 0)
        log<software_critical, -1>({__FILE__, __LINE__});

    if (threadStart(m_fgThread, m_fgThreadInit, m_fgThreadID, m_fgThreadProp, m_fgThreadPrio, m_fgCpuset, "framegrabber", this, fgThreadStart) < 0)
    {
        return log<software_critical, -1>({__FILE__, __LINE__});
    }

    if (threadStart(m_swThread, m_swThreadInit, m_swThreadID, m_swThreadProp, m_swThreadPrio, m_swCpuset, "streamwriter", this, swThreadStart) < 0)
    {
        log<software_critical, -1>({__FILE__, __LINE__});
    }

    if (telemeterT::appStartup() < 0)
    {
        return log<software_error, -1>({__FILE__, __LINE__});
    }

    return 0;
}

int streamWriter::appLogic()
{

    // first do a join check to see if other threads have exited.
    // these will throw if the threads are really gone
    try
    {
        if (pthread_tryjoin_np(m_fgThread.native_handle(), 0) == 0)
        {
            log<software_error>({__FILE__, __LINE__, "framegrabber thread has exited"});
            return -1;
        }
    }
    catch (...)
    {
        log<software_error>({__FILE__, __LINE__, "streamwriter thread has exited"});
        return -1;
    }

    try
    {
        if (pthread_tryjoin_np(m_swThread.native_handle(), 0) == 0)
        {
            log<software_error>({__FILE__, __LINE__, "stream thread has exited"});
            return -1;
        }
    }
    catch (...)
    {
        log<software_error>({__FILE__, __LINE__, "streamwriter thread has exited"});
        return -1;
    }

    switch (m_writing)
    {
    case NOT_WRITING:
        state(stateCodes::READY);
        break;
    default:
        state(stateCodes::OPERATING);
    }

    if (state() == stateCodes::OPERATING)
    {
        if (telemeterT::appLogic() < 0)
        {
            log<software_error>({__FILE__, __LINE__});
            return 0;
        }
    }

    updateINDI();

    return 0;
}

int streamWriter::appShutdown()
{
    try
    {
        if (m_fgThread.joinable())
        {
            m_fgThread.join();
        }
    }
    catch (...)
    {
    }

    try
    {
        if (m_swThread.joinable())
        {
            m_swThread.join();
        }
    }
    catch (...)
    {
    }

    if (m_xrif)
    {
        xrif_delete(m_xrif);
        m_xrif = nullptr;
    }

    if (m_xrif_timing)
    {
        xrif_delete(m_xrif_timing);
        m_xrif_timing = nullptr;
    }

    telemeterT::appShutdown();

    return 0;
}

int streamWriter::initialize_xrif()
{
    xrif_error_t rv = xrif_new(&m_xrif);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle allocation or initialization error."});
    }

    if (m_compress)
    {
        rv = xrif_configure(m_xrif, XRIF_DIFFERENCE_PREVIOUS, XRIF_REORDER_BYTEPACK, XRIF_COMPRESS_LZ4);
        if (rv != XRIF_NOERROR)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle configuration error."});
        }
    }
    else
    {
        std::cerr << "not compressing . . . \n";
        rv = xrif_configure(m_xrif, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE);
        if (rv != XRIF_NOERROR)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle configuration error."});
        }
    }

    errno = 0;
    m_xrif_header = (char *)malloc(XRIF_HEADER_SIZE * sizeof(char));
    if (m_xrif_header == NULL)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "xrif header allocation failed."});
    }

    rv = xrif_new(&m_xrif_timing);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle allocation or initialization error."});
    }

    // m_xrif_timing->reorder_method = XRIF_REORDER_NONE;
    rv = xrif_configure(m_xrif_timing, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle configuration error."});
    }

    errno = 0;
    m_xrif_timing_header = (char *)malloc(XRIF_HEADER_SIZE * sizeof(char));
    if (m_xrif_timing_header == NULL)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "xrif header allocation failed."});
    }

    return 0;
}

int streamWriter::setSigSegvHandler()
{
    struct sigaction act;
    sigset_t set;

    act.sa_sigaction = &streamWriter::_handlerSigSegv;
    act.sa_flags = SA_SIGINFO;
    sigemptyset(&set);
    act.sa_mask = set;

    errno = 0;
    if (sigaction(SIGSEGV, &act, 0) < 0)
    {
        std::string logss = "Setting handler for SIGSEGV failed. Errno says: ";
        logss += strerror(errno);

        log<software_error>({__FILE__, __LINE__, errno, 0, logss});

        return -1;
    }

    errno = 0;
    if (sigaction(SIGBUS, &act, 0) < 0)
    {
        std::string logss = "Setting handler for SIGBUS failed. Errno says: ";
        logss += strerror(errno);

        log<software_error>({__FILE__, __LINE__, errno, 0, logss});

        return -1;
    }

    log<text_log>("Installed SIGSEGV/SIGBUS signal handler.", logPrio::LOG_DEBUG);

    return 0;
}

void streamWriter::_handlerSigSegv(int signum,
                                   siginfo_t *siginf,
                                   void *ucont)
{
    m_selfWriter->handlerSigSegv(signum, siginf, ucont);
}

void streamWriter::handlerSigSegv(int signum,
                                  siginfo_t *siginf,
                                  void *ucont)
{
    static_cast<void>(signum);
    static_cast<void>(siginf);
    static_cast<void>(ucont);

    m_restart = true;

    return;
}

int streamWriter::allocate_circbufs()
{
    if (m_rawImageCircBuff)
    {
        free(m_rawImageCircBuff);
    }

    errno = 0;
    m_rawImageCircBuff = (char *)malloc(m_width * m_height * m_typeSize * m_circBuffLength);

    if (m_rawImageCircBuff == NULL)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "buffer allocation failure"});
    }

    if (m_timingCircBuff)
    {
        free(m_timingCircBuff);
    }

    errno = 0;
    m_timingCircBuff = (uint64_t *)malloc(5 * sizeof(uint64_t) * m_circBuffLength);
    if (m_timingCircBuff == NULL)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, errno, 0, "buffer allocation failure"});
    }

    return 0;
}

int streamWriter::allocate_xrif()
{
    // Set up the image data xrif handle
    xrif_error_t rv;

    if (m_compress)
    {
        rv = xrif_configure(m_xrif, XRIF_DIFFERENCE_PREVIOUS, XRIF_REORDER_BYTEPACK, XRIF_COMPRESS_LZ4);
        if (rv != XRIF_NOERROR)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle configuration error."});
        }
    }
    else
    {
        std::cerr << "not compressing . . . \n";
        rv = xrif_configure(m_xrif, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE);
        if (rv != XRIF_NOERROR)
        {
            return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle configuration error."});
        }
    }

    rv = xrif_set_size(m_xrif, m_width, m_height, 1, m_writeChunkLength, m_dataType);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif_set_size error."});
    }

    rv = xrif_allocate_raw(m_xrif);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif_allocate_raw error."});
    }

    rv = xrif_allocate_reordered(m_xrif);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif_allocate_reordered error."});
    }

    // Set up the timing data xrif handle
    rv = xrif_configure(m_xrif_timing, XRIF_DIFFERENCE_NONE, XRIF_REORDER_NONE, XRIF_COMPRESS_NONE);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif handle configuration error."});
    }

    rv = xrif_set_size(m_xrif_timing, 5, 1, 1, m_writeChunkLength, XRIF_TYPECODE_UINT64);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif_set_size error."});
    }

    rv = xrif_allocate_raw(m_xrif_timing);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif_allocate_raw error."});
    }

    rv = xrif_allocate_reordered(m_xrif_timing);
    if (rv != XRIF_NOERROR)
    {
        return log<software_critical, -1>({__FILE__, __LINE__, 0, rv, "xrif_allocate_reordered error."});
    }

    return 0;
}

void streamWriter::fgThreadStart(streamWriter *o)
{
    o->fgThreadExec();
}

void streamWriter::fgThreadExec()
{
    m_fgThreadID = syscall(SYS_gettid);

    // Wait fpr the thread starter to finish initializing this thread.
    while (m_fgThreadInit == true && m_shutdown == 0)
    {
        sleep(1);
    }

    timespec missing_ts;

    IMAGE image;
    ino_t inode = 0; // The inode of the image stream file

    bool opened = false;

    while (m_shutdown == 0)
    {
        /* Initialize ImageStreamIO
         */
        opened = false;
        m_restart = false; // Set this up front, since we're about to restart.

        sem_t *sem{nullptr}; ///< The semaphore to monitor for new image data

        int logged = 0;
        while (!opened && !m_shutdown && !m_restart)
        {
            // b/c ImageStreamIO prints every single time, and latest version don't support stopping it yet, and that isn't thread-safe-able anyway
            // we do our own checks.  This is the same code in ImageStreamIO_openIm...
            int SM_fd;
            char SM_fname[200];
            ImageStreamIO_filename(SM_fname, sizeof(SM_fname), m_shmimName.c_str());
            SM_fd = open(SM_fname, O_RDWR);
            if (SM_fd == -1)
            {
                if (!logged)
                    log<text_log>("ImageStream " + m_shmimName + " not found (yet).  Retrying . . .", logPrio::LOG_NOTICE);
                logged = 1;
                sleep(1); // be patient
                continue;
            }

            // Found and opened,  close it and then use ImageStreamIO
            logged = 0;
            close(SM_fd);

            if (ImageStreamIO_openIm(&image, m_shmimName.c_str()) == 0)
            {
                if (image.md[0].sem <= m_semaphoreNumber) ///<\todo this isn't right--> isn't there a define in cacao to use?
                {
                    ImageStreamIO_closeIm(&image);
                    mx::sys::sleep(1); // We just need to wait for the server process to finish startup.
                }
                else
                {
                    opened = true;

                    char SM_fname[200];
                    ImageStreamIO_filename(SM_fname, sizeof(SM_fname), m_shmimName.c_str());

                    struct stat buffer;
                    int rv = stat(SM_fname, &buffer);

                    if (rv != 0)
                    {
                        log<software_critical>({__FILE__, __LINE__, errno, "Could not get inode for " + m_shmimName + ". Source process will need to be restarted."});
                        ImageStreamIO_closeIm(&image);
                        return;
                    }
                    inode = buffer.st_ino;
                }
            }
            else
            {
                mx::sys::sleep(1); // be patient
            }
        }

        if (m_restart)
            continue; // this is kinda dumb.  we just go around on restart, so why test in the while loop at all?

        if (m_shutdown || !opened)
        {
            if (!opened)
                return;

            ImageStreamIO_closeIm(&image);
            return;
        }

        // now get a good semaphore
        m_semaphoreNumber = ImageStreamIO_getsemwaitindex(&image, m_semaphoreNumber); // ask for semaphore we had before

        if (m_semaphoreNumber < 0)
        {
            log<software_critical>({__FILE__, __LINE__, "No valid semaphore found for " + m_shmimName + ". Source process will need to be restarted."});
            return;
        }

        log<software_info>({__FILE__, __LINE__, "got semaphore index " + std::to_string(m_semaphoreNumber) + " for " + m_shmimName});

        ImageStreamIO_semflush(&image, m_semaphoreNumber);

        sem = image.semptr[m_semaphoreNumber]; ///< The semaphore to monitor for new image data

        m_dataType = image.md[0].datatype;
        m_typeSize = ImageStreamIO_typesize(m_dataType);
        m_width = image.md[0].size[0];
        m_height = image.md[0].size[1];
        size_t length;
        if (image.md[0].naxis == 3)
        {
            length = image.md[0].size[2];
        }
        else
        {
            length = 1;
        }
        std::cerr << "connected"
                  << " " << m_width << "x" << m_height << "x" << (int)m_dataType << " (" << m_typeSize << ")" << std::endl;

        // Now allocate the circBuffs
        if (allocate_circbufs() < 0)
            return; // will cause shutdown!

        // And allocate the xrifs
        if (allocate_xrif() < 0)
            return; // Will cause shutdown!

        uint8_t atype;
        size_t snx, sny, snz;

        uint64_t curr_image; // The current cnt1 index
        m_currImage = 0;
        m_currChunkStart = 0;
        m_nextChunkStart = 0;

        // Initialized curr_image ...
        if (image.md[0].naxis > 2)
        {
            curr_image = image.md[0].cnt1;
        }
        else
        {
            curr_image = 0;
        }

        uint64_t last_cnt0; // = ((uint64_t)-1);

        // so we can initialize last_cnt0 to avoid frame skip on startup
        if (image.cntarray)
        {
            last_cnt0 = image.cntarray[curr_image];
        }
        else
        {
            last_cnt0 = image.md[0].cnt0;
        }

        int cnt0flag = 0;

        bool restartWriting = false; // flag to prevent logging on a logging restart

        // This is the main image grabbing loop.
        while (!m_shutdown && !m_restart)
        {
            timespec ts;
            XWC_SEM_WAIT_TS_RETVOID(ts, m_semWaitSec, m_semWaitNSec);

            if(sem_timedwait(sem, &ts) == 0)
            {
                if (image.md[0].naxis > 2)
                {
                    curr_image = image.md[0].cnt1;
                }
                else
                {
                    curr_image = 0;
                }

                atype = image.md[0].datatype;
                snx = image.md[0].size[0];
                sny = image.md[0].size[1];
                if (image.md[0].naxis == 3)
                {
                    snz = image.md[0].size[2];
                }
                else
                {
                    snz = 1;
                }

                if (atype != m_dataType || snx != m_width || sny != m_height || snz != length)
                {
                    break; // exit the nearest while loop and get the new image setup.
                }

                if (m_shutdown || m_restart)
                {
                    break; // Check for exit signals
                }

                uint64_t new_cnt0;
                if (image.cntarray)
                {
                    new_cnt0 = image.cntarray[curr_image];
                }
                else
                {
                    new_cnt0 = image.md[0].cnt0;
                }

                std::cerr << "new_cnt0: " << new_cnt0 << "\n";
                
                ///\todo cleanup skip frame handling.
                if (new_cnt0 == last_cnt0) //<- this probably isn't useful really
                {
                    log<text_log>("semaphore raised but cnt0 has not changed -- we're probably getting behind", logPrio::LOG_WARNING);
                    ++cnt0flag;
                    if (cnt0flag > 10)
                    {
                        m_restart = true; // if we get here 10 times then something else is wrong.
                    }
                    continue;
                }

                if (new_cnt0 - last_cnt0 > 1) //<- this is what we want to check.
                {
                    log<text_log>("cnt0 changed by more than 1. Frame skipped.", logPrio::LOG_WARNING);
                }

                cnt0flag = 0;

                last_cnt0 = new_cnt0;

                char *curr_dest = m_rawImageCircBuff + m_currImage * m_width * m_height * m_typeSize;
                char *curr_src = (char *)image.array.raw + curr_image * m_width * m_height * m_typeSize;

                memcpy(curr_dest, curr_src, m_width * m_height * m_typeSize);

                uint64_t *curr_timing = m_timingCircBuff + 5 * m_currImage;

                if (image.cntarray)
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
                if (curr_timing[1] == 0)
                {

                    if (clock_gettime(CLOCK_REALTIME, &missing_ts) < 0)
                    {
                        log<software_critical>({__FILE__, __LINE__, errno, 0, "clock_gettime"});
                        return;
                    }

                    curr_timing[1] = missing_ts.tv_sec;
                    curr_timing[2] = missing_ts.tv_nsec;
                }

                // just set w-time to a-time if it's missing
                if (curr_timing[3] == 0)
                {
                    curr_timing[3] = curr_timing[1];
                    curr_timing[4] = curr_timing[2];
                }

                m_currImageTime = 1.0 * curr_timing[3] + (1.0 * curr_timing[4]) / 1e9;

                if (m_shutdown && m_writing == WRITING)
                {
                    m_writing = STOP_WRITING;
                }

                switch (m_writing)
                {
                    case START_WRITING:
                        if(!restartWriting)
                        {
                            m_currChunkStart = m_currImage;
                            m_nextChunkStart = (m_currImage / m_writeChunkLength) * m_writeChunkLength;
                            m_currChunkStartTime = m_currImageTime;
        
                            log<saving_start>({1, new_cnt0});
                        }
                        else
                        {
                            m_currChunkStart = m_currImage;
                            m_nextChunkStart = (m_currImage / m_writeChunkLength) * m_writeChunkLength;

                            if(m_currImage - m_nextChunkStart == m_writeChunkLength - 1)
                            {
                                m_nextChunkStart += m_writeChunkLength;
                            }

                            m_currChunkStartTime = m_currImageTime;

                            restartWriting = false;
                        }

                        m_writing = WRITING;

                        // fall through
                    case WRITING:
                        if(m_currImage - m_nextChunkStart == m_writeChunkLength - 1)
                        {
                            m_currSaveStart = m_currChunkStart;
                            m_currSaveStop = m_nextChunkStart + m_writeChunkLength;
                            m_currSaveStopFrameNo = new_cnt0;
    
                            std::cerr << __FILE__ << " " << __LINE__ << " WRITING " << m_currImage << " " 
                                                   << m_nextChunkStart << " " 
                                                    << (m_currImage - m_nextChunkStart == m_writeChunkLength - 1) << " "
                                                     << (m_currImageTime - m_currChunkStartTime > m_maxChunkTime) << " "
                                                      << new_cnt0 << "\n";

                            
                            // Now tell the writer to get going
                            if (sem_post(&m_swSemaphore) < 0)
                            {
                                log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                                return;
                            }
    
                            m_nextChunkStart = ((m_currImage + 1) / m_writeChunkLength) * m_writeChunkLength;
                            if (m_nextChunkStart >= m_circBuffLength)
                            {
                                m_nextChunkStart = 0;
                            }
    
                            m_currChunkStart = m_nextChunkStart;
                            m_currChunkStartTime = m_currImageTime;
                        }
                        else if(m_currImageTime - m_currChunkStartTime > m_maxChunkTime)
                        {
                            m_currSaveStart = m_currChunkStart;
                            m_currSaveStop = m_currImage + 1;
                            m_currSaveStopFrameNo = new_cnt0;
    
                            std::cerr << __FILE__ << " " << __LINE__ << " IMAGE TIME WRITING " << m_currImage << " " 
                                                   << m_nextChunkStart << " " 
                                                    << (m_currImage - m_nextChunkStart == m_writeChunkLength - 1) << " "
                                                     << (m_currImageTime - m_currChunkStartTime > m_maxChunkTime) << " "
                                                      << new_cnt0 << "\n";

                            // Now tell the writer to get going
                            if (sem_post(&m_swSemaphore) < 0)
                            {
                                log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                                return;
                            }
    
                            m_nextChunkStart = ((m_currImage + 1) / m_writeChunkLength) * m_writeChunkLength;
                            if (m_nextChunkStart >= m_circBuffLength)
                            {
                                m_nextChunkStart = 0;
                            }
    
                            m_currChunkStart = m_nextChunkStart;
                            m_currChunkStartTime = m_currImageTime;
                        }
                        break;
    
                    case STOP_WRITING:
                        m_currSaveStart = m_currChunkStart;
                        m_currSaveStop = m_currImage+1;
                        m_currSaveStopFrameNo = new_cnt0;
    
                        std::cerr << __FILE__ << " " << __LINE__ << " STOP_WRITING\n";
                        // Now tell the writer to get going
                        if (sem_post(&m_swSemaphore) < 0)
                        {
                            log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                            return;
                        }
                        restartWriting = false;
                        break;
    
                    default:
                        break;
                }

                ++m_currImage;
                if (m_currImage >= m_circBuffLength)
                {
                    m_currImage = 0;
                }
            }
            else
            {
                // If semaphore times-out or errors, we first cleanup any writing that needs to be done
                //we can also get here if a signal interrupts the sem wait which is triggered by INDI callbacks
                switch (m_writing)
                {
                    case WRITING:
                        // Here, if there is at least 1 image, we check for delta-time > m_maxChunkTime
                        //  then write
                        if ((m_currImage - m_nextChunkStart > 0) && (mx::sys::get_curr_time() - m_currChunkStartTime > m_maxChunkTime))
                        {
                            m_currSaveStart = m_currChunkStart;
                            m_currSaveStop = m_currImage;
                            m_currSaveStopFrameNo = last_cnt0;
    
                            std::cerr << __FILE__ << " " << __LINE__ << " TIMEOUT WRITING " << " " 
                                << m_currImage << " " << m_nextChunkStart << " " <<(m_currImage - m_nextChunkStart)  << " "
                                 << last_cnt0 << "\n";
                            // Now tell the writer to get going
                            if (sem_post(&m_swSemaphore) < 0)
                            {
                                log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                                return;
                            }
    
                            m_writing = START_WRITING;
                            restartWriting = true;

                            

                            /*m_currChunkStart = m_currImage;
                            m_nextChunkStart = ((m_currImage + 1) / m_writeChunkLength) * m_writeChunkLength;
                            if (m_nextChunkStart >= m_circBuffLength)
                            {
                                m_nextChunkStart = 0;
                            }
                            m_currChunkStartTime = m_currImageTime;*/
                        }
                        break;
                    case STOP_WRITING:
                        // If we timed-out while STOP_WRITING is set, we trigger a write.
                        m_currSaveStart = m_currChunkStart;
                        m_currSaveStop = m_currImage;
                        m_currSaveStopFrameNo = last_cnt0;
    
                        std::cerr << __FILE__ << " " << __LINE__ << " TIMEOUT STOP_WRITING\n";
                        // Now tell the writer to get going
                        if (sem_post(&m_swSemaphore) < 0)
                        {
                            log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                            return;
                        }
                        restartWriting = false;
                        break;
                    default:
                        break;
                }

                if (image.md[0].sem <= 0)
                {
                    break; // Indicates that the server has cleaned up.
                }

                // Check for why we timed out
                if (errno == EINTR)
                {
                    break; // This will indicate time to shutdown, loop will exit normally flags set.
                }

                // ETIMEDOUT just means we should wait more.
                // Otherwise, report an error.
                if (errno != ETIMEDOUT)
                {
                    log<software_error>({__FILE__, __LINE__, errno, "sem_timedwait"});
                    break;
                }

                // Check if the file has disappeared.
                int SM_fd;
                char SM_fname[200];
                ImageStreamIO_filename(SM_fname, sizeof(SM_fname), m_shmimName.c_str());
                SM_fd = open(SM_fname, O_RDWR);
                if (SM_fd == -1)
                {
                    m_restart = true;
                }
                close(SM_fd);

                // Check if the inode changed
                struct stat buffer;
                int rv = stat(SM_fname, &buffer);
                if (rv != 0)
                {
                    m_restart = true;
                }

                if (buffer.st_ino != inode)
                {
                    std::cerr << "Restarting due to inode . . . \n";
                    m_restart = true;
                }
            }
        }

        ///\todo might still be writing here, so must check
        // If semaphore times-out or errors, we first cleanup any writing that needs to be done
        if(m_writing == WRITING || m_writing == STOP_WRITING)
        {
            // Here, if there is at least 1 image, then write
            if ((m_currImage - m_nextChunkStart > 0))
            {
                m_currSaveStart = m_currChunkStart;
                m_currSaveStop = m_currImage;
                m_currSaveStopFrameNo = last_cnt0;
    
                m_writing = STOP_WRITING;

                std::cerr << __FILE__ << " " << __LINE__ << " WRITING ON RESTART " << last_cnt0 << "\n";
                // Now tell the writer to get going
                if (sem_post(&m_swSemaphore) < 0)
                {
                    log<software_critical>({__FILE__, __LINE__, errno, 0, "Error posting to semaphore"});
                    return;
                }
             }
             else
             {
                m_writing = NOT_WRITING;
             }

             
             while(m_writing != NOT_WRITING)
             {
                std::cerr << __FILE__ << " " << __LINE__ << " WAITING TO FINISH WRITING " << last_cnt0 << "\n";
                sleep(1);
             }
        }

        if (m_rawImageCircBuff)
        {
            free(m_rawImageCircBuff);
            m_rawImageCircBuff = 0;
        }

        if (m_timingCircBuff)
        {
            free(m_timingCircBuff);
            m_timingCircBuff = 0;
        }

        if (opened)
        {
            if (m_semaphoreNumber >= 0)
            {
                ///\todo is this release necessary with closeIM?
                image.semReadPID[m_semaphoreNumber] = 0; // release semaphore
            }
            ImageStreamIO_closeIm(&image);
            opened = false;
        }

    } // outer loop, will exit if m_shutdown==true

    ///\todo might still be writing here, so must check
    // One more check
    if (m_rawImageCircBuff)
    {
        free(m_rawImageCircBuff);
        m_rawImageCircBuff = 0;
    }

    if (m_timingCircBuff)
    {
        free(m_timingCircBuff);
        m_timingCircBuff = 0;
    }

    if (opened)
    {
        if (m_semaphoreNumber >= 0)
        {
            ///\todo is this release necessary with closeIM?
            image.semReadPID[m_semaphoreNumber] = 0; // release semaphore.
        }

        ImageStreamIO_closeIm(&image);
    }
}

void streamWriter::swThreadStart(streamWriter *s)
{
    s->swThreadExec();
}

void streamWriter::swThreadExec()
{
    m_swThreadID = syscall(SYS_gettid);

    // Wait fpr the thread starter to finish initializing this thread.
    while (m_swThreadInit == true && m_shutdown == 0)
    {
        sleep(1);
    }

    while (!m_shutdown)
    {
        while (!shutdown() && (!(state() == stateCodes::READY || state() == stateCodes::OPERATING)))
        {
            if (m_fname)
            {
                free(m_fname);
                m_fname = nullptr;
            }
            sleep(1);
        }

        if (shutdown())
        {
            break;
        }

        // This will happen after a reconnection, and could update m_shmimName, etc.
        if (m_fname == nullptr)
        {
            m_fnameBase = m_rawimageDir + "/" + m_outName + "_";

            m_fnameSz = m_fnameBase.size() + sizeof("YYYYMMDDHHMMSSNNNNNNNNN.xrif"); // the sizeof includes the \0
            m_fname = (char *)malloc(m_fnameSz);

            snprintf(m_fname, m_fnameSz, "%sYYYYMMDDHHMMSSNNNNNNNNN.xrif", m_fnameBase.c_str());
        }

        // at this point fname is not null.

        timespec ts;

        if (clock_gettime(CLOCK_REALTIME, &ts) < 0)
        {
            log<software_critical>({__FILE__, __LINE__, errno, 0, "clock_gettime"});

            free(m_fname);
            m_fname = nullptr;

            return; // will trigger a shutdown
        }

        mx::sys::timespecAddNsec(ts, m_semWaitNSec);

        if (sem_timedwait(&m_swSemaphore, &ts) == 0)
        {
            if (doEncode() < 0)
            {
                log<software_critical>({__FILE__, __LINE__, "error encoding data"});
                return;
            }
            // Otherwise, success, and we just go on.
        }
        else
        {
            // Check for why we timed out
            if (errno == EINTR)
            {
                continue; // This will probably indicate time to shutdown, loop will exit normally if flags set.
            }

            // ETIMEDOUT just means we should wait more.
            // Otherwise, report an error.
            if (errno != ETIMEDOUT)
            {
                log<software_error>({__FILE__, __LINE__, errno, "sem_timedwait"});
                break;
            }
        }
    } // outer loop, will exit if m_shutdown==true

    if (m_fname)
    {
        free(m_fname);
        m_fname = nullptr;
    }
}

int streamWriter::doEncode()
{
    if (m_writing == NOT_WRITING)
    {
        return 0;
    }

    recordSavingState(true);

    timespec tw0, tw1, tw2;

    clock_gettime(CLOCK_REALTIME, &tw0);

    // Record these to prevent a change in other thread
    uint64_t saveStart = m_currSaveStart;
    uint64_t saveStopFrameNo = m_currSaveStopFrameNo;
    size_t nFrames = m_currSaveStop - saveStart;
    size_t nBytes = m_width * m_height * m_typeSize;

    std::cerr << "nFrames: " << nFrames << "\n";

    // Configure xrif and copy image data -- this does no allocations
    int rv = xrif_set_size(m_xrif, m_width, m_height, 1, nFrames, m_dataType);
    if (rv != XRIF_NOERROR)
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>({__FILE__, __LINE__, 0, rv, "xrif set size error. DATA POSSIBLY LOST"});
    }

    rv = xrif_set_lz4_acceleration(m_xrif, m_lz4accel);
    if (rv != XRIF_NOERROR)
    {
        // This may just be out of range, it's only an error.
        log<software_error>({__FILE__, __LINE__, 0, rv, "xrif set LZ4 acceleration error."});
    }

    memcpy(m_xrif->raw_buffer, m_rawImageCircBuff + saveStart * nBytes, nFrames * nBytes);

    // Configure xrif and copy timing data -- no allocations
    rv = xrif_set_size(m_xrif_timing, 5, 1, 1, nFrames, XRIF_TYPECODE_UINT64);
    if (rv != XRIF_NOERROR)
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>({__FILE__, __LINE__, 0, rv, "xrif set size error. DATA POSSIBLY LOST."});
    }

    rv = xrif_set_lz4_acceleration(m_xrif_timing, m_lz4accel);
    if (rv != XRIF_NOERROR)
    {
        // This may just be out of range, it's only an error.
        log<software_error>({__FILE__, __LINE__, 0, rv, "xrif set LZ4 acceleration error."});
    }

    for (size_t nF = 0; nF < nFrames; ++nF)
    {
        std::cerr << "      " << (m_timingCircBuff + saveStart * 5 + nF * 5)[0] << "\n";
    }

    memcpy(m_xrif_timing->raw_buffer, m_timingCircBuff + saveStart * 5, nFrames * 5 * sizeof(uint64_t));

    rv = xrif_encode(m_xrif);
    if (rv != XRIF_NOERROR)
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>({__FILE__, __LINE__, 0, rv, "xrif encode error. DATA POSSIBLY LOST."});
    }

    rv = xrif_write_header(m_xrif_header, m_xrif);
    if (rv != XRIF_NOERROR)
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>({__FILE__, __LINE__, 0, rv, "xrif write header error. DATA POSSIBLY LOST."});
    }

    rv = xrif_encode(m_xrif_timing);
    if (rv != XRIF_NOERROR)
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>({__FILE__, __LINE__, 0, rv, "xrif encode error. DATA POSSIBLY LOST."});
    }

    rv = xrif_write_header(m_xrif_timing_header, m_xrif_timing);
    if (rv != XRIF_NOERROR)
    {
        // This is a big problem.  Report it as "ALERT" and go on.
        log<software_alert>({__FILE__, __LINE__, 0, rv, "xrif write header error. DATA POSSIBLY LOST"});
    }

    // Now break down the acq time of the first image in the buffer for use in file name
    tm uttime; // The broken down time.
    timespec *fts = (timespec *)(m_timingCircBuff + saveStart * 5 + 1);

    if (gmtime_r(&fts->tv_sec, &uttime) == 0)
    {
        // Yell at operator but keep going
        log<software_alert>({__FILE__, __LINE__, errno, 0, "gmtime_r error.  possible loss of timing information."});
    }

    // Available size = m_fnameSz-m_fnameBase.size(), rather than assuming sizeof("YYYYMMDDHHMMSSNNNNNNNNN"), in case we screwed up somewhere.
    rv = snprintf(m_fname + m_fnameBase.size(), m_fnameSz - m_fnameBase.size(), "%04i%02i%02i%02i%02i%02i%09i", uttime.tm_year + 1900,
                  uttime.tm_mon + 1, uttime.tm_mday, uttime.tm_hour, uttime.tm_min, uttime.tm_sec, static_cast<int>(fts->tv_nsec));

    if (rv != sizeof("YYYYMMDDHHMMSSNNNNNNNNN") - 1)
    {
        // Something is very wrong.  Keep going to try to get it on disk.
        log<software_alert>({__FILE__, __LINE__, errno, rv, "did not write enough chars to timestamp"});
    }

    // Cover up the \0 inserted by snprintf
    (m_fname + m_fnameBase.size())[23] = '.';

    clock_gettime(CLOCK_REALTIME, &tw1);

    FILE *fp_xrif = fopen(m_fname, "wb");
    if (fp_xrif == NULL)
    {
        // This is it.  If we can't write data to disk need to fix.
        log<software_alert>({__FILE__, __LINE__, errno, 0, "failed to open file for writing"});

        free(m_fname);
        m_fname = nullptr;

        return -1; // will trigger a shutdown
    }

    size_t bw = fwrite(m_xrif_header, sizeof(uint8_t), XRIF_HEADER_SIZE, fp_xrif);

    if (bw != XRIF_HEADER_SIZE)
    {
        log<software_alert>({__FILE__, __LINE__, errno, 0, "failure writing header to file.  DATA LOSS LIKELY. bytes = " + std::to_string(bw)});
        // We go on . . .
    }

    bw = fwrite(m_xrif->raw_buffer, sizeof(uint8_t), m_xrif->compressed_size, fp_xrif);

    if (bw != m_xrif->compressed_size)
    {
        log<software_alert>({__FILE__, __LINE__, errno, 0, "failure writing data to file.  DATA LOSS LIKELY. bytes = " + std::to_string(bw)});
    }

    bw = fwrite(m_xrif_timing_header, sizeof(uint8_t), XRIF_HEADER_SIZE, fp_xrif);

    if (bw != XRIF_HEADER_SIZE)
    {
        log<software_alert>({__FILE__, __LINE__, errno, 0, "failure writing timing header to file.  DATA LOSS LIKELY.  bytes = " + std::to_string(bw)});
    }

    bw = fwrite(m_xrif_timing->raw_buffer, sizeof(uint8_t), m_xrif_timing->compressed_size, fp_xrif);

    if (bw != m_xrif_timing->compressed_size)
    {
        log<software_alert>({__FILE__, __LINE__, errno, 0, "failure writing timing data to file. DATA LOSS LIKELY. bytes = " + std::to_string(bw)});
    }

    fclose(fp_xrif);

    clock_gettime(CLOCK_REALTIME, &tw2);

    double wt = ((double)tw2.tv_sec + ((double)tw2.tv_nsec) / 1e9) - ((double)tw1.tv_sec + ((double)tw1.tv_nsec) / 1e9);

    std::cerr << wt << "\n";

    recordSavingStats(true);

    if (m_writing == STOP_WRITING)
    {
        m_writing = NOT_WRITING;
        log<saving_stop>({0, saveStopFrameNo});
    }

    recordSavingState(true);

    return 0;

} // doEncode

INDI_NEWCALLBACK_DEFN(streamWriter, m_indiP_writing)
(const pcf::IndiProperty &ipRecv)
{
    INDI_VALIDATE_CALLBACK_PROPS(m_indiP_writing, ipRecv);

    if (!ipRecv.find("toggle"))
    {
        return 0;
    }

    if (ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off && (m_writing == WRITING || m_writing == START_WRITING))
    {
        m_writing = STOP_WRITING;
    }

    if (ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On && m_writing == NOT_WRITING)
    {
        m_writing = START_WRITING;
    }

    return 0;
}

void streamWriter::updateINDI()
{
    // Only update this if not changing
    if (m_writing == NOT_WRITING || m_writing == WRITING)
    {
        if (m_xrif && m_writing == WRITING)
        {
            indi::updateSwitchIfChanged(m_indiP_writing, "toggle", pcf::IndiElement::On, m_indiDriver, INDI_OK);
            indi::updateIfChanged(m_indiP_xrifStats, "ratio", m_xrif->compression_ratio, m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "encodeMBsec", m_xrif->encode_rate / 1048576.0, m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "encodeFPS", m_xrif->encode_rate / (m_width * m_height * m_typeSize), m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "differenceMBsec", m_xrif->difference_rate / 1048576.0, m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "differenceFPS", m_xrif->difference_rate / (m_width * m_height * m_typeSize), m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "reorderMBsec", m_xrif->reorder_rate / 1048576.0, m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "reorderFPS", m_xrif->reorder_rate / (m_width * m_height * m_typeSize), m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "compressMBsec", m_xrif->compress_rate / 1048576.0, m_indiDriver, INDI_BUSY);
            indi::updateIfChanged(m_indiP_xrifStats, "compressFPS", m_xrif->compress_rate / (m_width * m_height * m_typeSize), m_indiDriver, INDI_BUSY);
        }
        else
        {
            indi::updateSwitchIfChanged(m_indiP_writing, "toggle", pcf::IndiElement::Off, m_indiDriver, INDI_OK);
            indi::updateIfChanged(m_indiP_xrifStats, "ratio", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "encodeMBsec", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "encodeFPS", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "differenceMBsec", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "differenceFPS", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "reorderMBsec", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "reorderFPS", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "compressMBsec", 0.0, m_indiDriver, INDI_IDLE);
            indi::updateIfChanged(m_indiP_xrifStats, "compressFPS", 0.0, m_indiDriver, INDI_IDLE);
        }
    }
}

int streamWriter::checkRecordTimes()
{
    return telemeterT::checkRecordTimes(telem_saving_state());
}

int streamWriter::recordTelem(const telem_saving_state *)
{
    return recordSavingState(true);
}

int streamWriter::recordSavingState(bool force)
{
    static int16_t lastState = -1;
    static uint64_t currSaveStart = -1;

    int16_t state;
    if (m_writing == WRITING || m_writing == START_WRITING || m_writing == STOP_WRITING) // Changed from just writing 5/2024
        state = 1;
    else
        state = 0;

    if (state != lastState || m_currSaveStart != currSaveStart || force)
    {
        telem<telem_saving_state>({state, m_currSaveStart});

        lastState = state;
        currSaveStart = m_currSaveStart;
    }

    return 0;
}

int streamWriter::recordSavingStats(bool force)
{
    static uint32_t last_rawSize = -1;
    static uint32_t last_compressedSize = -1;
    static float last_encodeRate = -1;
    static float last_differenceRate = -1;
    static float last_reorderRate = -1;
    static float last_compressRate = -1;

    if (m_xrif->raw_size != last_rawSize || m_xrif->compressed_size != last_compressedSize || m_xrif->encode_rate != last_encodeRate || m_xrif->difference_rate != last_differenceRate ||
        m_xrif->reorder_rate != last_reorderRate || m_xrif->compress_rate != last_compressRate || force)
    {
        telem<telem_saving>({(uint32_t)m_xrif->raw_size, (uint32_t)m_xrif->compressed_size, (float)m_xrif->encode_rate, (float)m_xrif->difference_rate, (float)m_xrif->reorder_rate, (float)m_xrif->compress_rate});

        last_rawSize = m_xrif->raw_size;
        last_compressedSize = m_xrif->compressed_size;
        last_encodeRate = m_xrif->encode_rate;
        last_differenceRate = m_xrif->difference_rate;
        last_reorderRate = m_xrif->reorder_rate;
        last_compressRate = m_xrif->compress_rate;
    }

    return 0;
}

} // namespace app
} // namespace MagAOX

#endif
