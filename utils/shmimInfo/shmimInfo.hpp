/** \file shmimInfo.hpp
 * \brief The shmimInfo class declaration.
 *
 * \ingroup shmimInfo_files
 *
 * \author Codex
 */

#ifndef shmimInfo_hpp
#define shmimInfo_hpp

#include <ImageStreamIO/ImageStruct.h>
#include <ImageStreamIO/ImageStreamIO.h>

#include "../../libMagAOX/libMagAOX.hpp"

/** \defgroup shmimInfo shmimInfo: report shmim dimensions and measured frame rate
 * \brief Attaches to a shmim, reports its dimensions, and measures frame rate over a configurable number of frames.
 *
 * \ingroup utils
 *
 */

/** \defgroup shmimInfo_files shmimInfo Files
 * \ingroup shmimInfo
 */

/// Utility for reporting shmim dimensions and a measured frame rate.
/**
 * \ingroup shmimInfo
 */
class shmimInfo : public mx::app::application
{
  protected:
    /** \name Configurable Parameters
     * @{
     */

    std::string m_shmimName; ///< Name of the shmim to inspect.

    size_t m_nFrames{ 100 }; ///< Number of frame arrivals to time when measuring frame rate.

    double m_timeoutSec{ 1.0 }; ///< Timeout, in seconds, used for each frame-arrival wait.

    ///@}

    /** \name Stream State - Data
     * @{
     */

    IMAGE m_imageStream{}; ///< Attached ImageStreamIO handle for the shmim under test.

    bool m_opened{ false }; ///< Tracks whether `m_imageStream` currently owns an open attachment.

    ino_t m_inode{ 0 }; ///< Inode of the backing shmim file used to detect stream replacement while waiting.

    int m_semaphoreNumber{ 9 }; ///< Preferred semaphore index used to subscribe to frame arrivals.

    uint32_t m_width{ 0 }; ///< Stream size along the first axis.

    uint32_t m_height{ 1 }; ///< Stream size along the second axis, or 1 when absent.

    uint32_t m_depth{ 1 }; ///< Stream size along the third axis, or 1 when absent.

    ///@}

  public:
    /// Default constructor.
    shmimInfo();

    /// Destructor.
    ~shmimInfo() override;

    /// Define command-line and config-file options.
    void setupConfig() override;

    /// Load configured values into member state.
    void loadConfig() override;

    /// Connect to the shmim, report dimensions, and measure frame rate.
    int execute() override;

  protected:
    /// Open the shmim and cache its dimensions.
    int openStream();

    /// Close the shmim attachment if it is open.
    void closeStream();

    /// Wait for the configured number of frame arrivals and report the measured FPS.
    int measureFrames();

    /// Convert a pair of timespec timestamps to elapsed seconds.
    static double elapsedSeconds( const timespec &t0 /**< [in] starting timestamp. */,
                                  const timespec &t1 /**< [in] ending timestamp. */ );
};

#endif // shmimInfo_hpp
