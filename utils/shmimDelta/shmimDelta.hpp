/** \file shmimDelta.hpp
 * \brief The shmimDelta class declaration.
 *
 * \ingroup shmimDelta_files
 *
 * \author Codex
 */

#ifndef shmimDelta_hpp
#define shmimDelta_hpp

#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include <semaphore.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <time.h>
#include <vector>

#include "../../libMagAOX/libMagAOX.hpp"

/** \defgroup shmimDelta shmimDelta: measure semaphore-arrival deltas between two shmims
 * \brief Attaches to two shmims and reports the mean and rms delta between their semaphore arrivals.
 *
 * <a href="../handbook/utils/shmimDelta.html">Utility Documentation</a>
 *
 * \ingroup utils
 *
 */

/** \defgroup shmimDelta_files shmimDelta Files
 * \ingroup shmimDelta
 */

/// Utility for measuring the delta between two shmim semaphore arrival times.
/**
 * \ingroup shmimDelta
 */
class shmimDelta : public mx::app::application
{
  protected:
    /// Runtime state for one monitored shmim.
    struct streamState
    {
        /// One measured frame arrival.
        struct frameSample
        {
            uint64_t m_cnt0{ 0 }; ///< Frame counter sampled after the semaphore wake.

            timespec m_eventTime{}; ///< Frame timestamp used for delta calculation.

            timespec m_arrivalTime{}; ///< Local timestamp recorded after the semaphore wake.

            bool m_usedMetadataTime{ false }; ///< True when `m_eventTime` came from stream metadata.
        };

        std::string m_name; ///< Name of the shmim to monitor.

        IMAGE m_imageStream{}; ///< Attached ImageStreamIO handle for this shmim.

        bool m_opened{ false }; ///< Tracks whether `m_imageStream` owns an open attachment.

        ino_t m_inode{ 0 }; ///< Inode of the backing shmim file used to detect stream replacement.

        int m_semaphoreNumber{ 9 }; ///< Preferred semaphore index used to subscribe to frame arrivals.

        sem_t *m_semaphore{ nullptr }; ///< Semaphore selected by ImageStreamIO for this reader.

        uint32_t m_width{ 0 }; ///< Stream size along the first axis.

        uint32_t m_height{ 1 }; ///< Stream size along the second axis, or 1 when absent.

        uint32_t m_depth{ 1 }; ///< Stream size along the third axis, or 1 when absent.

        std::vector<frameSample> m_samples; ///< Unique frame samples recorded after semaphore wakes.

        frameSample m_referenceSample; ///< Synchronized frame sample used as the counter reference.

        bool m_haveReference{ false }; ///< True once `m_referenceSample` is valid.

        std::string m_errorMessage; ///< Error text recorded by the waiter thread on failure.

        int m_status{ 0 }; ///< Waiter-thread completion status, 0 on success and -1 on error.
    };

    /** \name Configurable Parameters
     * @{
     */

    std::string m_shmimName1; ///< Name of the first shmim, used as the reference arrival time.

    std::string m_shmimName2; ///< Name of the second shmim. Reported delta is shmimName2 minus shmimName1.

    size_t m_nFrames{ 100 }; ///< Number of paired semaphore arrivals to measure.

    double m_timeoutSec{ 1.0 }; ///< Timeout, in seconds, used for each frame-arrival wait.

    ///@}

    /** \name Stream State - Data
     * @{
     */

    streamState m_stream1; ///< State for `m_shmimName1`.

    streamState m_stream2; ///< State for `m_shmimName2`.

    ///@}

    /** \name Measurement - Data
     * @{
     */

    std::vector<double> m_deltaUsec; ///< Paired frame-arrival deltas in microseconds.

    ///@}

  public:
    /// Default constructor.
    shmimDelta();

    /// Destructor.
    ~shmimDelta() override;

    /// Define command-line and config-file options.
    void setupConfig() override;

    /// Load configured values into member state.
    void loadConfig() override;

    /// Connect to both shmims, collect semaphore arrivals, and report delta statistics.
    int execute() override;

    /// Convert a pair of timespec timestamps to elapsed microseconds.
    static double elapsedUsec( const timespec &t0 /**< [in] starting timestamp. */,
                               const timespec &t1 /**< [in] ending timestamp. */ );

    /// Calculate the arithmetic mean of a vector.
    static double mean( const std::vector<double> &values /**< [in] values to average. */ );

    /// Calculate the rms scatter about a supplied mean.
    static double rms( const std::vector<double> &values /**< [in] values to calculate rms for. */,
                       double                     meanValue /**< [in] arithmetic mean of `values`. */ );

  protected:
    /// Open one shmim and cache its dimensions.
    int openStream( streamState &stream /**< [in,out] stream state to open. */ );

    /// Close one shmim attachment if it is open.
    void closeStream( streamState &stream /**< [in,out] stream state to close. */ );

    /// Close both shmim attachments if open.
    void closeStreams();

    /// Select and flush the semaphore used to monitor one stream.
    int prepareSemaphore( streamState &stream /**< [in,out] stream state to prepare. */ );

    /// Collect semaphore-arrival timestamps from both streams and report statistics.
    int measureDeltas();

    /// Synchronize the stream counter references before collecting the full measurement.
    int synchronizeStreams();

    /// Build delta samples by pairing matching counter advances from the synchronized references.
    size_t pairByReferenceCounter();

    /// Build frame-to-frame delta samples for one stream.
    size_t calculateFrameDeltas( const streamState   &stream,     /**< [in] stream state to inspect. */
                                 std::vector<double> &frameDeltas /**< [out] frame-to-frame deltas in microseconds. */
    ) const;

    /// Calculate a sample's frame-counter advance from its synchronized reference.
    bool counterAdvance( uint64_t                       &advance,  /**< [out] frame-counter advance. */
                         const streamState::frameSample &sample,   /**< [in] sample to compare. */
                         const streamState::frameSample &reference /**< [in] synchronized reference sample. */
    ) const;

    /// Thread entry point for collecting arrival timestamps from one stream.
    static void waitThreadStart( shmimDelta  *app /**< [in] owning application instance. */,
                                 streamState *stream /**< [in,out] stream state to monitor. */ );

    /// Collect all configured frame-arrival timestamps for one stream.
    int waitFrames( streamState &stream /**< [in,out] stream state to monitor. */ );

    /// Wait for one frame arrival and record its timestamp.
    int waitForFrame( streamState &stream /**< [in,out] stream state to monitor. */,
                      size_t       frameNumber /**< [in] one-based frame number used for error reporting. */ );

    /// Record the latest frame metadata after a semaphore wake.
    int recordLatestSample( streamState &stream /**< [in,out] stream state to sample. */ );

    /// Get the latest frame counter and timestamp from a stream.
    streamState::frameSample latestSample( const streamState &stream /**< [in] stream state to sample. */ ) const;

    /// Check whether a timespec is non-zero.
    static bool validTime( const timespec &ts /**< [in] timestamp to inspect. */ );

    /// Check whether one timestamp is at or after another timestamp.
    static bool timeAtOrAfter( const timespec &ts, /**< [in] timestamp to inspect. */
                               const timespec &reference /**< [in] reference timestamp. */ );

    /// Check whether a stream disappeared or was replaced after a wait error.
    bool streamChanged( const streamState &stream /**< [in] stream state to check. */ ) const;

    /// Fill a timespec with the absolute timeout deadline for sem_timedwait.
    int setWaitDeadline( timespec &deadline /**< [out] absolute realtime wait deadline. */ ) const;
};

#endif // shmimDelta_hpp
