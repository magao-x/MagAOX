/** \file dm.hpp
 * \brief The MagAO-X generic deformable mirror controller.
 *
 * \ingroup app_files
 */

#ifndef dm_hpp
#define dm_hpp

#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/math/floatUtils.hpp>

#include "dmTiming.hpp"
#include "dmCompletion.hpp"
#include "dmSaturation.hpp"
#include "shmimMonitor.hpp"
#include "../../logger/types/telem_dmtimings.hpp"
#include "../../logger/types/telem_fgtimings.hpp"

namespace MagAOX
{
namespace app
{
namespace dev
{

/// Get the ImageStreamIO type code, or zero for an unsupported type.
template <typename typeT>
constexpr uint8_t ImageStreamTypeCode()
{
    return 0;
}

/// Get the ImageStreamIO code for single-precision DM commands.
template <>
constexpr uint8_t ImageStreamTypeCode<float>()
{
    return _DATATYPE_FLOAT;
}

/// Get the ImageStreamIO code for double-precision DM commands.
template <>
constexpr uint8_t ImageStreamTypeCode<double>()
{
    return _DATATYPE_DOUBLE;
}

/** MagAO-X generic deformable mirror controller
 *
 *
 * The derived class `derivedT` must meet the following requirements:
 * - Must be a MagAOXApp<true>
 *
 * - Must be a dev::shmimMonitor<derivedT>
 *
 * - Must NOT call dev::shmimMonitor<derivedT>::setupConfig and dev::shmimMonitor<derivedT>::loadConfig.  These are
 *   handled by this class.
 *
 * - Must expose the following interface
 *   \code
 *       int initDM();
 *       int commandDM(void * cmd);
 *       int zeroDM();
 *      int releaseDM();
 *    \endcode
 *   Each of the above functions should return 0 on success, and -1 on an error.
 *
 * - This class must be declared a friend in the derived class, like so:
 *   \code
 *       friend class dev::dm<derivedT,realT>;
 *   \endcode
 *
 * - Must contain the following typedef:
 *   \code
 *       typedef dev::dm<derivedT, realT> dmT;
 *   \endcode
 *
 * - Calls to this class's `setupConfig`, `loadConfig`, `appStartup`, `appLogic`, `appShutdown`, and `updateINDI`
 *   functions must be placed in the derived class's functions of the same name. For convenience the
 *   following macros are defined to provide error checking:
 *   \code
 *       DM_SETUP_CONFIG( cfig )
 *       DM_LOAD_CONFIG( cfig )
 *       DM_APP_STARTUP
 *       DM_APP_LOGIC
 *       DM_UPDATE_INDI
 *       DM_APP_SHUTDOWN
 *   \endcode
 *
 *
 * \ingroup appdev
 */
template <class derivedT, typename realT>
class dm
{

    typedef mx::verbose::vvv verboseT;

  protected:
    /** \name Configurable Parameters - Data
     * @{
     */

    std::string m_calibPath; ///< The path to this DM's calibration files.
    std::string m_flatPath;  ///< The path to this DM's flat files (usually the same as calibPath)
    std::string m_testPath;  ///< The path to this DM's test files (default is calibPath/tests).

    std::string m_actMaskPath; ///< The file name of the actuator mask for this DM

    std::string m_flatDefault; ///< The file name of the this DM's default flat command. Path and extension will be
                               ///< ignored and can be omitted.
    std::string m_testDefault; ///< The file name of the this DM's default test command. Path and extension will be
                               ///< ignored and can be omitted.

    std::string m_shmimFlat;    ///< The name of the shmim stream to write the flat to.
    std::string m_shmimTest;    ///< The name of the shmim stream to write the test to.
    std::string m_shmimSat;     ///< The name of the shmim stream to write the saturation map to.
    std::string m_shmimSatPerc; ///< The name of the shmim stream to write the saturation percentage map to.

    int m_satAvgInt{ 100 }; ///< The time in milliseconds to accumulate saturation over.

    int m_satThreadPrio{ 0 }; ///< Priority of the saturation thread.  Usually ok to be 0.

    std::string m_shmimShape; ///< The name of the shmim stream to write the desaturated true shape to.
    std::string m_shmimDelta; ///< The name of the shmim stream to write the desaturated delta command to.
    std::string m_shmimDiff;  ///< The name of the shmim stream to write the difference to.

    std::string m_completionStreamName; ///< Optional shared reader-completion stream; empty disables publication.

    std::string m_readerReleaseStreamName; ///< Optional last-channel-read signal, independent of backend success.

    uint32_t m_dmWidth{ 0 };  ///< The width of the images in the stream
    uint32_t m_dmHeight{ 0 }; ///< The height of the images in the stream

    static constexpr uint8_t m_dmDataType = ImageStreamTypeCode<realT>(); ///< The ImageStreamIO type code.

    float m_percThreshold{ 0.98 }; ///<  Threshold on percentage of frames an actuator is saturated over an interval.

    /// Fraction of actuators exceeding percThreshold required to count an excessive-saturation interval.
    float m_intervalSatThreshold{ 0.50 };

    /// Number of consecutive excessive-saturation intervals required before opening configured loops.
    int m_intervalSatCountThreshold{ 10 };

    std::vector<std::string> m_satTriggerDevice; ///< Device(s) with a toggle switch to toggle on saturation trigger.

    /// Toggle-switch properties to clear after excessive saturation, paired with satTriggerDevice.
    std::vector<std::string> m_satTriggerProperty;

    ///@}

    std::string m_calibRelDir; ///< The directory relative to the calibPath.  Set this before calling
                               ///< dm<derivedT,realT>::loadConfig().

    int m_numChannels{ 0 }; ///< The number of dmcomb channels found as part of allocation.

    /// Owned mappings of the numbered main channels; auxiliary sources are opened only when clearing.
    std::vector<mx::improc::milkImage<realT> *> m_channels;

    std::map<std::string, std::string> m_flatCommands; ///< Map of flat file name to full path
    std::string                        m_flatCurrent;  ///< The name of the current flat command

    mx::improc::eigenImage<realT> m_flatCommand;         ///< Data storage for the flat command
    bool                          m_flatLoaded{ false }; ///< Flag indicating whether a flat is loaded in memory

    IMAGE m_flatImageStream;  ///< The ImageStreamIO shared memory buffer for the flat.
    bool  m_flatSet{ false }; ///< Flag indicating whether the flat command has been set.

    /// Shared-memory actuator mask used to restrict loaded flat and test commands.
    mx::improc::milkImage<realT> m_actMask;

    std::map<std::string, std::string> m_testCommands; ///< Map of test file name to full path
    std::string                        m_testCurrent;  ///< Name of the currently loaded test command.

    mx::improc::eigenImage<realT> m_testCommand;         ///< Data storage for the test command
    bool                          m_testLoaded{ false }; ///< Flag indicating whether a test command is loaded in memory

    IMAGE m_testImageStream;  ///< The ImageStreamIO shared memory buffer for the test.
    bool  m_testSet{ false }; ///< Flag indicating whether the test command has been set.

    /// Instantaneous actuator saturation flags populated by derivedT::commandDM().
    mx::improc::eigenImage<uint8_t> m_instSatMap;

    /// Per-actuator saturation counts accumulated during m_satAvgInt milliseconds.
    mx::improc::eigenImage<uint16_t> m_accumSatMap;

    /// Per-actuator fraction of saturated commands during the latest accumulation interval.
    mx::improc::eigenImage<float> m_satPercMap;

    IMAGE m_satImageStream;     ///< The ImageStreamIO shared memory buffer for the sat map.
    IMAGE m_satPercImageStream; ///< The ImageStreamIO shared memory buffer for the sat percentage map.

    int  m_overSatAct{ 0 };         ///< Actuators exceeding the saturation percentage threshold in the latest interval.
    int  m_intervalSatExceeds{ 0 }; ///< Consecutive accumulation intervals exceeding the saturation threshold.
    bool m_intervalSatTrip{ 0 };    ///< Pending request to open configured loops after persistent saturation.

    mx::improc::milkImage<realT> m_outputShape; ///< The true output shape after saturation.

    std::vector<std::string> m_deltaChannels; ///< The names of channels which are treated as delta commands

    std::vector<size_t> m_deltas;    ///< Indices of the channels which are delta commands
    std::vector<size_t> m_notDeltas; ///< Indices of the channels which are not delta commands

    mx::improc::eigenImage<realT> m_totalFlat;  ///< the total of all non-delta channels
    mx::improc::eigenImage<realT> m_totalDelta; ///< the total of all delta channels

    mx::improc::eigenImage<realT> m_stagedDelta; ///< Private delta result retained until submission finalization.

    mx::improc::eigenImage<realT> m_stagedDiff; ///< Private difference result calculated from the staged delta.

    bool m_deltaStaged{ false }; ///< Current callback has unpublished private delta/difference products.

    dev::dmSaturation m_pipelineSaturation; ///< Lossless pending saturation payload for asynchronous publication.

    std::mutex m_pipelineSatMutex; ///< Protects pipeline saturation geometry and consumer window state.

    mx::improc::eigenImage<uint64_t> m_pipelineSatWindow; ///< Saturation-thread-owned pipeline accumulation.

    std::atomic<bool> m_pipelineSatEnabled{ false }; ///< Published after pipeline saturation storage is ready.

    uint64_t m_pipelineSatGeneration{ 0 }; ///< Geometry/reset generation protected by m_pipelineSatMutex.

    mx::improc::milkImage<realT> m_outputDelta; ///< The true output delta command after saturation.
    mx::improc::milkImage<realT>
        m_outputDiff; ///< The difference between command and true delta command after saturation.

    /** \name Saturation Thread Data
     * This thread processes the saturation maps
     * @{
     */

    sem_t m_satSemaphore; ///< Semaphore used to tell the saturation thread to run.

    bool m_satThreadInit{ true }; ///< Synchronizer for thread startup, to allow priority setting to finish.

    pid_t m_satThreadID{ 0 }; ///< The ID of the saturation thread.

    pcf::IndiProperty m_satThreadProp; ///< The property to hold the saturation thread details.

    std::thread m_satThread; ///< A separate thread for the actual saturation processing

    ///@}

  public:
    /// Destructor
    /** deallocates the m_channels vector
     *
     */
    ~dm();

    /** \name Configurable Parameters
     * @{
     */

    /// Get the calibration directory.
    /** \returns the current value of m_calibPath */
    const std::string &calibPath() const;

    /// Get the flat-command directory.
    /** \returns the current value of m_flatPath */
    const std::string &flatPath() const;

    /// Get the test-command directory.
    /** \returns the current value of m_testPath */
    const std::string &testPath() const;

    /// Get the default flat-command name.
    /** \returns the current value of m_flatDefault */
    const std::string &flatDefault() const;

    /// Get the default test-command name.
    /** \returns the current value of m_testDefault */
    const std::string &testDefault() const;

    /// Get the flat-command stream name.
    /** \returns the current value of m_shmimFlat */
    const std::string &shmimFlat() const;

    /// Get the test-command stream name.
    /** \returns the current value of m_shmimTest */
    const std::string &shmimTest() const;

    /// Get the saturation-map stream name.
    /** \returns the current value of m_shmimSat */
    const std::string &shmimSat() const;

    /// Get the stream name for saturation percentage
    /**
     * \returns the current value of m_shmimSatPerc
     */
    const std::string &shmimSatPerc() const;

    /// Get the saturation accumulation interval
    /**
     * \returns the current value of m_satAvgInt
     */
    int satAvgInt() const;

    /// Get the saturation thread priority
    /**
     * \returns the current value of m_satThreadPrio
     */
    int satThreadPrio() const;

    /// Get the desaturated output-shape stream name.
    /** \returns the current value of m_shmimShape */
    const std::string &shmimShape() const;

    /// Get the desaturated delta-command stream name.
    /** \returns the current value of m_shmimDelta */
    const std::string &shmimDelta() const;

    /// Get the DM Width
    /**
     * \returns the current value of m_dmWidth
     */
    uint32_t dmWidth() const;

    /// Get the DM Height
    /**
     * \returns the current value of m_dmHeight
     */
    uint32_t dmHeight() const;

    /// Get the DM data type
    /**
     * \returns the current value of m_dmDataType
     */
    uint8_t dmDataType() const;

    /// Get the saturation percentage threshold
    /**
     * \returns the current value of m_percThreshold
     */
    float percThreshold() const;

    /// Get the interval saturation threshold
    /**
     * \returns the current value of m_intervalSatThreshold
     */
    float intervalSatThreshold() const;

    /// Get the interval saturation count threshold
    /**
     * \returns the current value of m_intervalSatCountThreshold
     */
    int intervalSatCountThreshold() const;

    /// Get the saturation trigger device(s)
    /**
     * \returns the current value of m_satTriggerDevice
     */
    const std::vector<std::string> &satTriggerDevice() const;

    /// Get the saturation trigger property(ies)
    /**
     * \returns the current value of m_satTriggerProperty
     */
    const std::vector<std::string> &satTriggerProperty() const;

    /// Get the calibration directory relative to the application calibration root.
    const std::string &calibRelDir() const;

    /// Get the discovered number of numbered main channels.
    int numChannels() const;

    /// Get the instantaneous actuator saturation flags.
    const mx::improc::eigenImage<uint8_t> &instSatMap() const;

    /// Get the accumulated actuator saturation counts.
    const mx::improc::eigenImage<uint16_t> &accumSatMap() const;

    /// Get each actuator's saturation fraction over the accumulation interval.
    const mx::improc::eigenImage<float> &satPercMap() const;

    /// Get the configured names of delta-command channels.
    const std::vector<std::string> &deltaChannels() const;

    /// Get indices of channels treated as fixed rather than delta commands.
    const std::vector<size_t> &notDeltas() const;

    /// Get the combined non-delta command calculated by makeDelta().
    const mx::improc::eigenImage<float> &totalFlat() const;

    ///@}

    /// Setup the configuration system
    /**
      * This should be called in `derivedT::setupConfig` as
      * \code
        dm<derivedT,realT>::setupConfig(config);
        \endcode
      * with appropriate error checking.
      */
    int setupConfig( mx::app::appConfigurator &config /**< [out] the derived classes configurator*/ );

    /// load the configuration system results
    /**
      * This should be called in `derivedT::loadConfig` as
      * \code
        dm<derivedT,realT>::loadConfig(config);
        \endcode
      * with appropriate error checking.
      */
    int loadConfig( mx::app::appConfigurator &config /**< [in] the derived classes configurator*/ );

    /// Startup function
    /**
      * This should be called in `derivedT::appStartup` as
      * \code
        dm<derivedT,realT>::appStartup();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int appStartup();

    /// DM application logic
    /** This should be called in `derivedT::appLogic` as
      * \code
        dm<derivedT,realT>::appLogic();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int appLogic();

    /// DM shutdown
    /** This should be called in `derivedT::appShutdown` as
      * \code
        dm<derivedT,realT>::appShutdown();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int appShutdown();

    /// DM Poweroff
    /** This should be called in `derivedT::onPowerOff` as
      * \code
        dm<derivedT,realT>::onPowerOff();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int onPowerOff();

    /// DM Poweroff Updates
    /** This should be called in `derivedT::whilePowerOff` as
      * \code
        dm<derivedT,realT>::whilePowerOff();
        \endcode
      * with appropriate error checking.
      *
      * \returns 0 on success
      * \returns -1 on error, which is logged.
      */
    int whilePowerOff();

    /// Find the DM comb channels
    /** Opens numbered main channels and classifies their delta-command membership.
     * Auxiliary and other suffixed streams are excluded.
     */
    int findDMChannels();

    /// Called after shmimMonitor connects to the dmXXdisp stream.  Checks for proper size.
    /**
     * \returns 0 on success
     * \returns -1 if incorrect size or data type in stream.
     */
    int allocate( const dev::shmimT &sp /**< [in] Monitored stream tag. */ );

    /// Submit a monitored command, calculate delta products and publish timing/optional reader completion.
    int processImage( void              *curr_src /**< [in] Current combined command pixels. */,
                      const dev::shmimT &sp /**< [in] Monitored stream tag. */ );

    /// Admit a callback without waiting on an already selected shared input; synchronous owners always admit.
    bool admitCommand();

    /// Return whether this callback requires private products until backend finalization.
    bool deferredCommand() const;

    /// Drain a pending submission and retain a prior processing error; synchronous owners need no join.
    int finalizeCommand( int processingResult /**< [in] Earlier processing result, negative on failure. */ ) noexcept;

    /// Release callback ownership after output, timing and completion finalization.
    void retireCommand() noexcept;

    /// Private products and fixed provenance owned by one asynchronous command until final publication.
    struct PipelineCommand
    {
        MagAOX::dmTiming::Event       timing;     ///< Detached event; one stage owns mutations at a time.
        MagAOX::dmCompletion::Record  completion; ///< Frozen input identity captured before any external reads.
        mx::improc::eigenImage<realT> shape;      ///< Clipped shape belonging to this command.
        mx::improc::eigenImage<realT> delta;      ///< Private delta awaiting successful submission.
        mx::improc::eigenImage<realT> diff;       ///< Private difference awaiting successful submission.
        std::vector<uint8_t>          saturation; ///< Per-command actuator-grid flags retained through finalization.
        long                          saturatedCount{ 0 };  ///< Number of saturated mapped actuators.
        bool                          deltaStaged{ false }; ///< Private delta and difference are ready for publication.
        int                           preparedStatus{ 0 }; ///< Preparation result independent of the actual SDK result.
        int64_t dispatchNs{ -1 }; ///< Monitor handoff boundary, or unavailable for an unsent command.
    };

    /// Allocate one job's private products while command owners are drained.
    void allocatePipelineCommand( PipelineCommand &command /**< [out] Reusable command storage. */ );

    /// Capture a genuine command's frozen identity and detach its receipt timing on the monitor.
    /** A negative return rejects this source observation; it does not establish a backend fault. */
    int capturePipelineCommand( PipelineCommand &command /**< [out] Exclusively leased command. */,
                                void            *pixels /**< [in] Combined source selected by the monitor. */ );

    /// Mark one exclusively owned job without modifying another thread's current event.
    static void pipelineMark( PipelineCommand         &command /**< [in,out] Exclusively owned command. */,
                              MagAOX::dmTiming::Marker marker /**< [in] Endpoint to capture. */ );

    /// Stage channel-dependent products using the command's private clipped shape.
    int stagePipelineCommand( PipelineCommand &command /**< [in,out] Monitor-owned command. */ );

    /// Publish final external-read clearance from the monitor, retaining the original identity.
    /** Source advancement withholds reader/full provenance without changing successful preparation status. */
    void releasePipelineCommand( PipelineCommand &command /**< [in,out] Prepared command. */ );

    /// Publish successful private products and frozen completion on the sole finalizer.
    int finishPipelineCommand(
        PipelineCommand          &command /**< [in,out] Fully prepared command with an SDK result. */,
        MagAOX::dmTiming::Outcome outcome /**< [in] Actual SDK outcome or no-send. */,
        int64_t                   backendResult /**< [in] Raw SDK return code, or unavailable. */,
        int64_t                   sdkStartNs /**< [in] SDK entry, or -1 for an unsent command. */,
        int64_t                   sdkEndNs /**< [in] Actual SDK return, or -1 for an unsent command. */,
        bool                      submitted /**< [in] Actual successful SDK submission independent of publication. */,
        int64_t                   sdkReadyNs = -1 /**< [in] Worker available before its next wait, or unavailable. */ );

    /// Route monitor observations without writing the finalizer-owned timing queue or completion publisher.
    void pipelineShmimMonitorTiming( const dev::shmimT &tag /**< [in] Main DM monitor tag. */,
                                     shmimMonitorEvent  event /**< [in] Receipt/attachment observation. */ );

    /// Discard an unadmitted receipt; the derived controller separately counts admission rejections.
    void discardPipelineCommand();

    /// Calls derived()->initDM()
    /**
     * \returns 0 on success
     * \returns -1 on error from derived()->initDM()
     */
    int baseInitDM();

    /// Calls derived()->releaseDM() and then zeros main/auxiliary channels and the saturation map.
    /** This is called by the relevant INDI callback
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int baseReleaseDM();

    /// Check the flats directory and update the list of flats if anything changes
    /** This is called once per appLogic and whilePowerOff loops.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int checkFlats();

    /// Load a flat file
    /** Uses the target argument for lookup in m_flatCommands to find the path
     * and loads the command in the local memory.  Calls setFlat if the flat
     * is currently set.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int loadFlat( const std::string &target /**< [in] the name of the flat to load */ );

    /// Send the current flat command to the DM
    /** Writes the command to the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setFlat( bool update = false /**< [in] If true, this is an update rather than a new set*/ );

    /// Zero the flat command on the DM
    /** Writes a 0 array the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int zeroFlat();

    /// Check the tests directory and update the list of tests if anything changes
    /** This is called once per appLogic and whilePowerOff loops.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int checkTests();

    /// Load a test file
    /** Uses the target argument for lookup in m_testCommands to find the path
     * and loads the command in the local memory.  Calls setTest if the test
     * is currently set.
     */
    int loadTest( const std::string &target /**< [in] Name used to look up the test-command file. */ );

    /// Send the current test command to the DM
    /** Writes the command to the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int setTest();

    /// Zero the test command on the DM
    /** Writes a 0 array the designated shmim.
     *
     * \returns 0 on success
     * \returns -1 on error
     */
    int zeroTest();

    /// Zero all numbered main channels and their existing, compatible `_aux` sources.
    /** Missing auxiliary sources are normal. Incompatible sources are left unchanged
     * and reported as errors after the main channels are cleared. This does not coordinate with writers;
     * commands written after the clear are the writer's responsibility. Only the last
     * main channel's semaphore is posted, unless nosem is true.
     *
     * \returns 0 on success
     * \returns \<0 on an error
     */
    int zeroAll( bool nosem = false /**< [in] [optional] if true then the semaphore
                                                         is not raised after zeroing all channels*/
    );

    /// Calculate the delta command from the output shape.
    int makeDelta();

    /// Read channels into private preallocated delta products without publishing shared pixels or notifications.
    int stageDelta();

    /// Calculate private products using one command's clipped shape and monitor-owned summation scratch.
    int stageDelta( Eigen::Ref<const mx::improc::eigenImage<realT>> shape /**< [in] Frozen clipped command shape. */,
                    mx::improc::eigenImage<realT>                  &delta /**< [out] Preallocated private delta. */,
                    mx::improc::eigenImage<realT>                  &diff /**< [out] Preallocated private difference. */,
                    bool &staged /**< [out] Whether both private products are ready. */ );

    /// Publish previously staged products after successful backend finalization, without rereading channels.
    int publishDelta();

    /// Clear the saturation maps and zero the shared memory.
    /**
     * \returns 0 on success
     * \returns -1 on error
     */
    int clearSat();

  protected:
    /** \name Saturation Thread Functions
     * This thread processes the saturation maps
     * @{
     */

    /// Thread starter, called by MagAOXApp::threadStart on thread construction.  Calls satThreadExec.
    static void satThreadStart( dm *d /**< [in] a pointer to a dm instance (normally this) */ );

    /// Execute saturation processing
    void satThreadExec();

    /// Trigger loop openings because of excessive saturation
    void intervalSatTrip();

    ///@}

  protected:
    /** \name INDI
     *
     *@{
     */
  protected:
    // declare our properties

    pcf::IndiProperty m_indiP_flat; ///< Property used to set and report the current flat

    pcf::IndiProperty m_indiP_init;    ///< Request to initialize the DM hardware.
    pcf::IndiProperty m_indiP_zero;    ///< Request to send the hardware zero command.
    pcf::IndiProperty m_indiP_release; ///< Request to release the DM hardware and clear commands.

    pcf::IndiProperty m_indiP_flats;     ///< INDI Selection switch containing the flat files.
    pcf::IndiProperty m_indiP_flatShmim; ///< Publish the shmim being used for the flat
    pcf::IndiProperty m_indiP_setFlat;   ///< INDI toggle switch to set the current flat.

    pcf::IndiProperty m_indiP_tests;     ///< INDI Selection switch containing the test pattern files.
    pcf::IndiProperty m_indiP_testShmim; ///< Publish the shmim being used for the test command
    pcf::IndiProperty m_indiP_setTest;   ///< INDI toggle switch to set the current test pattern.

    pcf::IndiProperty m_indiP_zeroAll; ///< Request to clear main channels and auxiliary command sources.

  public:
    /// The static callback function to be registered for initializing the DM.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_init( void *app,                      /**< [in] a pointer to this, will be
                                                                              static_cast-ed to derivedT.*/
                                    const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the
                                                                              the new property request.*/
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_init( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                    the new property request.*/
    );

    /// Dispatch the request to zero the DM hardware.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_zero( void *app,                      /**< [in] a pointer to this, will be
                                                                              static_cast-ed to derivedT.*/
                                    const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                              the new property request.*/
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_zero(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the
        the new property request.*/ );

    /// Dispatch the request to release the DM hardware and clear its commands.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_release( void *app,                      /**< [in] a pointer to this, will be
                                                                                 static_cast-ed to derivedT.*/
                                       const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                                 the new property request.*/
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_release( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                             the new property request.*/
    );

    /// The static callback function to be registered for selecting the flat file
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_flats( void *app,                      /**< [in] a pointer to this, will be
                                                                               static_cast-ed to derivedT.*/
                                     const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                               the new property request.*/
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_flats(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the new property request.*/ );

    /// The static callback function to be registered for setting the flat
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_setFlat( void                    *app,   /**< [in] Pointer to this, cast to derivedT. */
                                       const pcf::IndiProperty &ipRecv /**< [in] New INDI property request. */
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_setFlat(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the new property request.*/ );

    /// The static callback function to be registered for selecting the test file
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_tests( void                    *app,   /**< [in] Pointer to this, cast to derivedT. */
                                     const pcf::IndiProperty &ipRecv /**< [in] New INDI property request. */
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_tests(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the new property request.*/ );

    /// The static callback function to be registered for setting the test shape
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_setTest( void *app,                      /**< [in] a pointer to this, will be
                                                                                 static_cast-ed to derivedT.*/
                                       const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with
                                                                                 the new property request.*/
    );

    /// The callback called by the static version, to actually process the new request.
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_setTest( const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the
                                                                       the new property request.*/
    );

    /// The static callback function to be registered for zeroing all channels
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    static int st_newCallBack_zeroAll( void                    *app,   /**< [in] Pointer to this, cast to derivedT. */
                                       const pcf::IndiProperty &ipRecv /**< [in] New INDI property request. */
    );

    /// The callback for the zeroAll toggle switch, called by the static version
    /**
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int newCallBack_zeroAll(
        const pcf::IndiProperty &ipRecv /**< [in] the INDI property sent with the new property request.*/ );

    /// Update the INDI properties for this device controller
    /** You should call this once per main loop.
     * It is not called automatically.
     *
     * \returns 0 on success.
     * \returns -1 on error.
     */
    int updateINDI();

    ///@}

  protected:
    /** \name Timing - Data
     * @{ */
    double m_timingWindow{ 2.0 }; ///< Trailing timing window in seconds; zero disables capture.

    size_t m_timingCapacity{ 20000 }; ///< Queue and consumer-window event cap; zero disables capture.

    std::string m_timingDeliveryClockName{
        "none" }; ///< Explicit input-producer clock; none preserves receipt-only capture.

    MagAOX::dmTiming::PublicationClock m_timingDeliveryClock{
        MagAOX::dmTiming::PublicationClock::none }; ///< Resolved producer clock for opt-in delivery measurements.

    double m_timingDeliveryBracketUs{ 1 }; ///< Largest accepted producer-clock capture bracket, microseconds.

    dev::dmTiming m_dmTiming; ///< Single-producer capture and consumer-owned timing statistics.

    uint64_t m_timingLastRecord{ 0 }; ///< Summary sequence last emitted by this instance.

    bool m_timingRecorded{ false }; ///< Whether this instance has emitted a summary yet.

    pcf::IndiProperty m_indiP_dmTiming; ///< Receipt/send cadence and paired-latency statistics.

    pcf::IndiProperty m_indiP_dmStages; ///< Named controller-stage statistics, in seconds.

    pcf::IndiProperty m_indiP_dmTimingQuality; ///< Timing coverage and quality counters.

    pcf::IndiProperty
        m_indiP_dmTimingDelivery; ///< Selected producer clock and lifetime delivery-qualification coverage.

    pcf::IndiProperty m_indiP_dmTimingState; ///< Endpoint/clock definitions and measurement availability.

    /// @}

  public:
    /** \name Timing
     * @{ */

    /// Resolve an explicitly selected input-producer clock without assuming that every stream uses CLOCK_ISIO.
    static bool
    parseDeliveryClock( const std::string                  &name /**< [in] none, isio, realtime or tai. */,
                        MagAOX::dmTiming::PublicationClock &clock /**< [out] Resolved clock on success. */ );

    /// Observe monitor events on the sole command producer thread.
    void shmimMonitorTiming( const dev::shmimT &tag /**< [in] Main DM monitor tag. */,
                             shmimMonitorEvent  event /**< [in] Receipt/attachment observation. */ );

    /// Capture a controller phase marker without logging or allocating.
    void dmTimingMark( MagAOX::dmTiming::Marker marker /**< [in] Controller endpoint to capture. */ );

    /// Record backend submission status without changing command handling.
    void dmTimingOutcome( MagAOX::dmTiming::Outcome outcome /**< [in] Observed submission/callback result. */ );

    /// Retain the actual backend result for both timing and immutable reader-release diagnostics.
    void dmBackendResult( MagAOX::dmTiming::Outcome outcome /**< [in] Actual backend submission outcome. */,
                          int64_t result /**< [in] Raw backend return code; independent of callback failure. */ );

    /// Refresh the consumer-owned statistics outside the command thread.
    void updateDMTimings();

    /// Get the cached consumer-owned timing summary.
    const MagAOX::dmTiming::Summary &dmTimingSummary() const;

    /// Record the cached detailed summary and valid compatible timing record.
    int recordDMTimings( bool force = false /**< [in] Emit even when the summary is unchanged. */ );

    /// @}

  protected:
    /** \name Reader Completion - Data
     * @{ */

    dev::dmCompletion m_dmCompletion; ///< Command-thread completion publisher, independent of droppable telemetry.

    dev::dmCompletion
        m_dmReaderRelease; ///< Last-read publisher with immutable identity and later SDK-result enrichment.

    bool m_completionReceipt{ false }; ///< The next callback follows a genuine semaphore receipt, not initial data.

    MagAOX::dmTiming::Outcome m_completionOutcome{
        MagAOX::dmTiming::Outcome::unknown }; ///< Backend result even when timing is disabled.

    int64_t m_completionBackendResult{ std::numeric_limits<int64_t>::min() }; ///< Raw result, or unavailable sentinel.

    pcf::IndiProperty m_indiP_dmCompletion; ///< Ordinary-thread latest completion state and diagnostic duration.

    pcf::IndiProperty m_indiP_dmCompletionStream; ///< Completion stream identity and software-only endpoint contract.

    pcf::IndiProperty m_indiP_dmReaderRelease; ///< Latest last-read endpoint and independent SDK result.

    pcf::IndiProperty m_indiP_dmReaderReleaseStream; ///< Separate reader-release stream identity and contract.

    /// @}

  public:
    /** \name Reader Completion
     * @{ */

    /// Register optional read-only completion properties without changing controller scheduling.
    int setupDMCompletionINDI();

    /// Publish a coherent latest completion snapshot outside the command thread.
    void updateDMCompletionINDI();

    /// @}

  protected:
    /// Register read-only timing statistics and endpoint/quality descriptions.
    int setupDMTimingINDI();

    /// Register the optional last-read endpoint and independent backend result properties.
    int setupDMReaderReleaseINDI();

    /// Refresh last-read diagnostics outside the command thread.
    void updateDMReaderReleaseINDI();

  private:
    /// Retire every admitted callback and drain an unexpected exceptional exit before releasing its storage.
    struct CommandRetirement
    {
        dm &m_owner; ///< Generic controller retaining the callback's owned buffers.

        bool m_finalized{ false }; ///< The ordinary callback path has collected its backend result.

        /// Retain the admitted controller until the callback leaves scope.
        explicit CommandRetirement( dm &owner /**< [in] Admitted controller. */ ) noexcept;

        /// Drain any unfinished backend operation, then release whole-callback ownership.
        ~CommandRetirement() noexcept;
    };

    /// Emit timing records for owners with the standard telemetry interface.
    template <class ownerT>
    static auto timingRecord( ownerT                          &owner /**< [in] Concrete controller. */,
                              const MagAOX::dmTiming::Summary &summary /**< [in] Coherent cached summary. */,
                              int                              priority /**< [in] Preferred overload selector. */ )
        -> decltype( owner.template telem<logger::telem_dmtimings>( { summary } ), int() );

    /// Retain compatibility with software DM owners without a telemeter.
    static int timingRecord( derivedT                        &owner /**< [in] Owner without telemetry support. */,
                             const MagAOX::dmTiming::Summary &summary /**< [in] Unrecorded summary. */,
                             long                             priority /**< [in] Fallback overload selector. */ );

    /// Get the concrete controller used by the CRTP interface.
    derivedT &derived();
};

template <class derivedT, typename realT>
derivedT &dm<derivedT, realT>::derived()
{
    return *static_cast<derivedT *>( this );
}

template <class derivedT, typename realT>
dm<derivedT, realT>::CommandRetirement::CommandRetirement( dm &owner ) noexcept : m_owner( owner )
{
}

template <class derivedT, typename realT>
dm<derivedT, realT>::CommandRetirement::~CommandRetirement() noexcept
{
    if( !m_finalized )
    {
        m_owner.derived().finalizeCommand( -1 );
    }

    m_owner.derived().retireCommand();
}

template <class derivedT, typename realT>
bool dm<derivedT, realT>::admitCommand()
{
    return true;
}

template <class derivedT, typename realT>
bool dm<derivedT, realT>::deferredCommand() const
{
    return false;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::finalizeCommand( int processingResult ) noexcept
{
    return processingResult;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::retireCommand() noexcept
{
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::allocatePipelineCommand( PipelineCommand &command )
{
    command.shape.resize( m_dmWidth, m_dmHeight );
    command.shape.setZero();
    command.delta.resize( m_dmWidth, m_dmHeight );
    command.diff.resize( m_dmWidth, m_dmHeight );
    command.saturation.assign( size_t( m_dmWidth ) * m_dmHeight, 0 );
    command.deltaStaged    = false;
    command.preparedStatus = 0;
    command.saturatedCount = 0;
    command.dispatchNs     = -1;

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_pipelineSatMutex );

        if( !m_pipelineSatEnabled.load( std::memory_order_relaxed ) || m_pipelineSatWindow.rows() != m_dmWidth ||
            m_pipelineSatWindow.cols() != m_dmHeight )
        {
            m_pipelineSaturation.configure( command.saturation.size() );
            m_pipelineSatWindow.resize( m_dmWidth, m_dmHeight );
            m_pipelineSatWindow.setZero();
            ++m_pipelineSatGeneration;
            m_pipelineSatEnabled.store( true, std::memory_order_release );
        }
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::capturePipelineCommand( PipelineCommand &command, void *pixels )
{
    const bool genuine     = m_completionReceipt;
    m_completionReceipt    = false;
    command.completion     = dev::dmCompletion::captureRecord( &derived().m_imageStream, genuine );
    command.preparedStatus = 0;
    command.deltaStaged    = false;
    command.saturatedCount = 0;
    command.dispatchNs     = -1;
    m_dmReaderRelease.begin( &derived().m_imageStream, genuine );

    if( !m_dmTiming.active() )
    {
        m_dmTiming.beginEvent( false );
    }

    m_dmTiming.mark( MagAOX::dmTiming::Marker::entry );
    if( command.completion.inputInode != 0 && command.completion.inputCounter != 0 )
    {
        m_dmTiming.inputCounter( command.completion.inputCounter );
    }

    if( command.completion.publicationSec > 0 && command.completion.publicationNsec >= 0 &&
        command.completion.publicationNsec < 1000000000 &&
        command.completion.publicationSec <=
            ( std::numeric_limits<int64_t>::max() - command.completion.publicationNsec ) / 1000000000 )
    {
        m_dmTiming.inputPublication( command.completion.publicationSec * 1000000000 +
                                     command.completion.publicationNsec );
    }

    command.timing = m_dmTiming.detachEvent();

    if( pixels == nullptr || command.completion.state != MagAOX::dmCompletion::State::busy )
    {
        command.preparedStatus = -1;
        return -1;
    }

    if( m_dmReaderRelease.enabled() )
    {
        const auto &readers = m_dmReaderRelease.attempt();

        if( readers.inputInode != command.completion.inputInode ||
            readers.inputCounter != command.completion.inputCounter ||
            readers.publicationSec != command.completion.publicationSec ||
            readers.publicationNsec != command.completion.publicationNsec )
        {
            command.preparedStatus = -1;
            return -1;
        }
    }

    return 0;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::pipelineMark( PipelineCommand &command, MagAOX::dmTiming::Marker marker )
{
    if( command.timing.sequence != 0 )
    {
        dev::dmTiming::markEvent( command.timing, marker, dev::dmTiming::monotonicNowNs() );
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::stagePipelineCommand( PipelineCommand &command )
{
    if( command.preparedStatus < 0 )
    {
        return command.preparedStatus;
    }

    pipelineMark( command, MagAOX::dmTiming::Marker::deltaStart );
    pipelineMark( command, MagAOX::dmTiming::Marker::stagingStart );
    command.preparedStatus = stageDelta( command.shape, command.delta, command.diff, command.deltaStaged );
    pipelineMark( command, MagAOX::dmTiming::Marker::stagingEnd );
    return command.preparedStatus;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::releasePipelineCommand( PipelineCommand &command )
{
    const auto current  = dev::dmCompletion::captureRecord( &derived().m_imageStream, true );
    const bool coherent = command.completion.state == MagAOX::dmCompletion::State::busy &&
                          current.state == MagAOX::dmCompletion::State::busy &&
                          current.inputInode == command.completion.inputInode &&
                          current.inputCounter == command.completion.inputCounter &&
                          current.publicationSec == command.completion.publicationSec &&
                          current.publicationNsec == command.completion.publicationNsec;

    if( !coherent )
    {
        // A newer input revokes provenance; it does not invalidate successful private preparation or SDK work.
        command.completion.state = MagAOX::dmCompletion::State::unqualified;
    }

    m_dmReaderRelease.release( &derived().m_imageStream, command.preparedStatus, coherent );

    if( m_dmReaderRelease.enabled() && !MagAOX::dmCompletion::readersQualified( m_dmReaderRelease.attempt() ) )
    {
        // The publisher performs another identity observation; keep either observation's loss nonfatal.
        command.completion.state = MagAOX::dmCompletion::State::unqualified;
    }

    if( command.preparedStatus >= 0 )
    {
        if( m_dmReaderRelease.enabled() && command.timing.sequence != 0 )
        {
            dev::dmTiming::markEvent(
                command.timing, MagAOX::dmTiming::Marker::readRelease, m_dmReaderRelease.attempt().finishedNs );
        }
        else
        {
            pipelineMark( command, MagAOX::dmTiming::Marker::readRelease );
        }
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::finishPipelineCommand( PipelineCommand          &command,
                                                MagAOX::dmTiming::Outcome outcome,
                                                int64_t                   backendResult,
                                                int64_t                   sdkStartNs,
                                                int64_t                   sdkEndNs,
                                                bool                      submitted,
                                                int64_t                   sdkReadyNs )
{
    using MagAOX::dmTiming::Marker;
    using MagAOX::dmTiming::Outcome;
    int result = command.preparedStatus;

    if( command.timing.sequence != 0 )
    {
        if( sdkStartNs > 0 && sdkEndNs >= sdkStartNs )
        {
            dev::dmTiming::importSdkEvent(
                command.timing, sdkStartNs, sdkEndNs, outcome, command.dispatchNs, -1, -1, sdkReadyNs );
        }
        else
        {
            dev::dmTiming::outcomeEvent( command.timing, outcome );
        }

        command.timing.submitted = submitted;
    }

    if( outcome != Outcome::success )
    {
        result = -1;
    }

    if( result >= 0 )
    {
        try
        {
            pipelineMark( command, Marker::publicationStart );

            if( command.deltaStaged )
            {
                m_outputDelta = command.delta;
                m_outputDiff  = command.diff;
            }

            pipelineMark( command, Marker::publicationEnd );
            pipelineMark( command, Marker::deltaEnd );

            if( command.saturation.size() != size_t( m_instSatMap.size() ) ||
                !m_pipelineSaturation.add( command.saturation.data(), command.saturation.size() ) )
            {
                result = -1;
            }
            else
            {
                std::copy( command.saturation.begin(), command.saturation.end(), m_instSatMap.data() );
                pipelineMark( command, Marker::postStart );
                result = sem_post( &m_satSemaphore );
                pipelineMark( command, Marker::postEnd );
            }
        }
        catch( const std::exception &error )
        {
            result = -1;
            derivedT::template log<software_critical>( { std::string( "DM pipeline publication: " ) + error.what() } );
        }
        catch( ... )
        {
            result = -1;
            derivedT::template log<software_critical>( { "unknown exception publishing DM pipeline command" } );
        }
    }

    const auto callbackOutcome = result < 0 && outcome != Outcome::noSend ? Outcome::failed : outcome;

    if( command.timing.sequence != 0 )
    {
        dev::dmTiming::outcomeEvent( command.timing, callbackOutcome );
        command.timing.submitted = submitted;
        pipelineMark( command, Marker::finish );
    }

    command.completion.backendResult = backendResult;
    m_dmCompletion.publishCompleted( command.completion, result, callbackOutcome );

    if( command.timing.sequence != 0 )
    {
        m_dmTiming.submitEvent( command.timing );
    }

    return result;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::discardPipelineCommand()
{
    static_cast<void>( m_dmTiming.detachEvent() );
    m_completionReceipt = false;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::pipelineShmimMonitorTiming( const dev::shmimT &tag, shmimMonitorEvent event )
{
    static_cast<void>( tag );

    if( event == shmimMonitorEvent::receipt || event == shmimMonitorEvent::initial )
    {
        discardPipelineCommand();
        m_dmTiming.beginEvent( event == shmimMonitorEvent::receipt );
        m_completionReceipt = event == shmimMonitorEvent::receipt;
    }
    else
    {
        discardPipelineCommand();

        if( event == shmimMonitorEvent::connected )
        {
            m_dmTiming.newSession();
        }

        m_dmReaderRelease.reset();
    }
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::shmimMonitorTiming( const dev::shmimT &tag, shmimMonitorEvent event )
{
    static_cast<void>( tag );

    if( event == shmimMonitorEvent::connected )
    {
        m_dmTiming.newSession();
        m_completionReceipt = false;
        m_dmCompletion.reset();
        m_dmReaderRelease.reset();
    }
    else if( event == shmimMonitorEvent::disconnected )
    {
        m_completionReceipt = false;
        m_dmCompletion.reset();
        m_dmReaderRelease.reset();
    }
    else if( event == shmimMonitorEvent::receipt || event == shmimMonitorEvent::initial )
    {
        m_dmTiming.beginEvent( event == shmimMonitorEvent::receipt );
        m_completionReceipt = event == shmimMonitorEvent::receipt;
    }
    else if( event == shmimMonitorEvent::discarded )
    {
        m_dmTiming.outcome( MagAOX::dmTiming::Outcome::rejected );
        m_dmTiming.finishEvent();
        m_completionReceipt = false;
        m_dmCompletion.reset();
        m_dmReaderRelease.reset();
    }
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::dmTimingMark( MagAOX::dmTiming::Marker marker )
{
    m_dmTiming.mark( marker );
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::dmTimingOutcome( MagAOX::dmTiming::Outcome outcome )
{
    m_completionOutcome = outcome;
    m_dmTiming.outcome( outcome );
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::dmBackendResult( MagAOX::dmTiming::Outcome outcome, int64_t result )
{
    m_completionBackendResult = result;
    dmTimingOutcome( outcome );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setupDMCompletionINDI()
{
    if( m_completionStreamName.empty() )
    {
        return setupDMReaderReleaseINDI();
    }

    m_indiP_dmCompletion = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_dmCompletion.setDevice( derived().configName() );
    m_indiP_dmCompletion.setName( "dm_completion" );
    m_indiP_dmCompletion.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_dmCompletion.setState( pcf::IndiProperty::Idle );

    for( const auto *name : { "state",
                              "session",
                              "epoch",
                              "sequence",
                              "inputInode",
                              "inputCounter",
                              "processingUs",
                              "ageSec",
                              "outcome",
                              "qualified" } )
    {
        m_indiP_dmCompletion.add( pcf::IndiElement( name ) );
    }

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmCompletion ) < 0 )
    {
        return -1;
    }

    m_indiP_dmCompletionStream = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_dmCompletionStream.setDevice( derived().configName() );
    m_indiP_dmCompletionStream.setName( "dm_completion_stream" );
    m_indiP_dmCompletionStream.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_dmCompletionStream.setState( pcf::IndiProperty::Idle );
    m_indiP_dmCompletionStream.add( pcf::IndiElement( "name" ) );
    m_indiP_dmCompletionStream.add( pcf::IndiElement( "meaning" ) );
    m_indiP_dmCompletionStream["name"] = m_completionStreamName;
    m_indiP_dmCompletionStream["meaning"] =
        std::string( "application channel reads finished; hardware settling is not measured" );

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmCompletionStream ) < 0 )
    {
        return -1;
    }

    return setupDMReaderReleaseINDI();
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::updateDMCompletionINDI()
{
    updateDMReaderReleaseINDI();

    if( !m_dmCompletion.enabled() || !derived().m_indiDriver )
    {
        return;
    }

    const auto record    = m_dmCompletion.snapshot();
    const auto now       = dmTiming::monotonicNowNs();
    const bool qualified = MagAOX::dmCompletion::qualified( record );
    indi::updateIfChanged<double>(
        m_indiP_dmCompletion,
        { "state",
          "session",
          "epoch",
          "sequence",
          "inputInode",
          "inputCounter",
          "processingUs",
          "ageSec",
          "outcome",
          "qualified" },
        { double( record.state ),
          double( record.session ),
          double( record.epoch ),
          double( record.sequence ),
          double( record.inputInode ),
          double( record.inputCounter ),
          record.startedNs > 0 && record.finishedNs >= record.startedNs
              ? ( record.finishedNs - record.startedNs ) / 1000.0
              : -1,
          record.finishedNs > 0 && now >= record.finishedNs ? ( now - record.finishedNs ) / 1e9 : -1,
          double( record.outcome ),
          qualified ? 1.0 : 0.0 },
        derived().m_indiDriver,
        qualified ? pcf::IndiProperty::Ok : pcf::IndiProperty::Idle );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setupDMReaderReleaseINDI()
{
    if( m_readerReleaseStreamName.empty() )
    {
        return 0;
    }

    m_indiP_dmReaderRelease = pcf::IndiProperty( pcf::IndiProperty::Number );
    m_indiP_dmReaderRelease.setDevice( derived().configName() );
    m_indiP_dmReaderRelease.setName( "dm_reader_release" );
    m_indiP_dmReaderRelease.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_dmReaderRelease.setState( pcf::IndiProperty::Idle );

    for( const auto *name : { "state",
                              "session",
                              "epoch",
                              "sequence",
                              "inputInode",
                              "inputCounter",
                              "processingUs",
                              "ageSec",
                              "outcome",
                              "backendResult",
                              "qualified" } )
    {
        m_indiP_dmReaderRelease.add( pcf::IndiElement( name ) );
    }

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmReaderRelease ) < 0 )
    {
        return -1;
    }

    m_indiP_dmReaderReleaseStream = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_dmReaderReleaseStream.setDevice( derived().configName() );
    m_indiP_dmReaderReleaseStream.setName( "dm_reader_release_stream" );
    m_indiP_dmReaderReleaseStream.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_dmReaderReleaseStream.setState( pcf::IndiProperty::Idle );
    m_indiP_dmReaderReleaseStream.add( pcf::IndiElement( "name" ) );
    m_indiP_dmReaderReleaseStream.add( pcf::IndiElement( "meaning" ) );
    m_indiP_dmReaderReleaseStream["name"] = m_readerReleaseStreamName;
    m_indiP_dmReaderReleaseStream["meaning"] =
        std::string( "channel reads finished; SDK result independent; not backend readiness or mirror settling" );

    return derived().registerIndiPropertyReadOnly( m_indiP_dmReaderReleaseStream );
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::updateDMReaderReleaseINDI()
{
    if( !m_dmReaderRelease.enabled() || !derived().m_indiDriver )
    {
        return;
    }

    const auto record   = m_dmReaderRelease.snapshot();
    const auto now      = dmTiming::monotonicNowNs();
    const bool released = MagAOX::dmCompletion::readersQualified( record );
    indi::updateIfChanged<double>(
        m_indiP_dmReaderRelease,
        { "state",
          "session",
          "epoch",
          "sequence",
          "inputInode",
          "inputCounter",
          "processingUs",
          "ageSec",
          "outcome",
          "backendResult",
          "qualified" },
        { double( record.state ),
          double( record.session ),
          double( record.epoch ),
          double( record.sequence ),
          double( record.inputInode ),
          double( record.inputCounter ),
          record.startedNs > 0 && record.finishedNs >= record.startedNs
              ? ( record.finishedNs - record.startedNs ) / 1000.0
              : -1,
          record.finishedNs > 0 && now >= record.finishedNs ? ( now - record.finishedNs ) / 1e9 : -1,
          double( record.outcome ),
          double( record.backendResult ),
          released ? 1.0 : 0.0 },
        derived().m_indiDriver,
        released ? pcf::IndiProperty::Ok : pcf::IndiProperty::Idle );
}

template <class derivedT, typename realT>
bool dm<derivedT, realT>::parseDeliveryClock( const std::string &name, MagAOX::dmTiming::PublicationClock &clock )
{
    using Clock = MagAOX::dmTiming::PublicationClock;

    if( name == "none" )
    {
        clock = Clock::none;
    }
    else if( name == "realtime" || ( name == "isio" && CLOCK_ISIO == CLOCK_REALTIME ) )
    {
        clock = Clock::realtime;
    }
#ifdef CLOCK_TAI
    else if( name == "tai" || ( name == "isio" && CLOCK_ISIO == CLOCK_TAI ) )
    {
        clock = Clock::tai;
    }
#endif
    else
    {
        return false;
    }

    return true;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::updateDMTimings()
{
    m_dmTiming.update();
}

template <class derivedT, typename realT>
const MagAOX::dmTiming::Summary &dm<derivedT, realT>::dmTimingSummary() const
{
    return m_dmTiming.summary();
}

template <class derivedT, typename realT>
template <class ownerT>
auto dm<derivedT, realT>::timingRecord( ownerT &owner, const MagAOX::dmTiming::Summary &summary, int priority )
    -> decltype( owner.template telem<logger::telem_dmtimings>( { summary } ), int() )
{
    static_cast<void>( priority );
    owner.template telem<logger::telem_dmtimings>( { summary } );

    if( summary.fgValid )
    {
        const auto &a  = summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::receiptInterval )];
        const auto &w  = summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sendInterval )];
        const auto &wa = summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::receiptSend )];
        owner.template telem<logger::telem_fgtimings>( { a.mean, a.stddev, w.mean, w.stddev, wa.mean, wa.stddev } );
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::timingRecord( derivedT &owner, const MagAOX::dmTiming::Summary &summary, long priority )
{
    static_cast<void>( owner );
    static_cast<void>( summary );
    static_cast<void>( priority );
    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::recordDMTimings( bool force )
{
    const auto &summary = m_dmTiming.summary();

    if( force || !m_timingRecorded || summary.sequence != m_timingLastRecord )
    {
        int rv = timingRecord( derived(), summary, 0 );

        if( rv < 0 )
        {
            return rv;
        }

        m_timingLastRecord = summary.sequence;
        m_timingRecorded   = true;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setupDMTimingINDI()
{
    derived().createROIndiNumber( m_indiP_dmTiming, "dm_timing" );

    for( const auto *name : { "receipt_fps",
                              "receipt_min",
                              "receipt_max",
                              "receipt_jitter",
                              "receipt_count",
                              "send_fps",
                              "send_min",
                              "send_max",
                              "send_jitter",
                              "send_count",
                              "receipt_send",
                              "receipt_send_jitter",
                              "receipt_send_min",
                              "receipt_send_max",
                              "receipt_send_count" } )
    {
        m_indiP_dmTiming.add( pcf::IndiElement( name ) );
        m_indiP_dmTiming[name] = -1.0;
    }

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmTiming ) < 0 )
    {
        return -1;
    }

    derived().createROIndiNumber( m_indiP_dmStages, "dm_timing_stages" );

    for( size_t i = MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::dispatch );
         i < MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::count );
         ++i )
    {
        std::string name = MagAOX::dmTiming::metricName( static_cast<MagAOX::dmTiming::Metric>( i ) );

        for( const auto *field : { "mean", "jitter", "min", "max", "count", "span" } )
        {
            std::string element = name + '_' + field;
            m_indiP_dmStages.add( pcf::IndiElement( element ) );
            m_indiP_dmStages[element] = -1.0;
        }
    }

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmStages ) < 0 )
    {
        return -1;
    }

    derived().createROIndiNumber( m_indiP_dmTimingQuality, "dm_timing_quality" );

    for( const auto *name : { "window",  "span",      "age",      "capacity",   "queued",      "session", "sequence",
                              "enabled", "valid",     "events",   "receipts",   "submitted",   "failed",  "partial",
                              "noSend",  "synthetic", "rejected", "duplicates", "counterGaps", "invalid", "dropped" } )
    {
        m_indiP_dmTimingQuality.add( pcf::IndiElement( name ) );
    }

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmTimingQuality ) < 0 )
    {
        return -1;
    }

    derived().createROIndiNumber( m_indiP_dmTimingDelivery, "dm_timing_delivery" );

    for( const auto *name : { "enabled",
                              "clock",
                              "bracketUs",
                              "attempted",
                              "qualified",
                              "unavailable",
                              "invalid",
                              "wideBracket",
                              "clockDiscontinuity",
                              "futurePublication",
                              "identityRejected" } )
    {
        m_indiP_dmTimingDelivery.add( pcf::IndiElement( name ) );
        m_indiP_dmTimingDelivery[name] = 0.0;
    }

    const auto &delivery                  = m_dmTiming.summary().delivery;
    m_indiP_dmTimingDelivery["enabled"]   = delivery.enabled ? 1.0 : 0.0;
    m_indiP_dmTimingDelivery["clock"]     = static_cast<double>( delivery.clock );
    m_indiP_dmTimingDelivery["bracketUs"] = delivery.bracketLimitNs * 0.001;

    if( derived().registerIndiPropertyReadOnly( m_indiP_dmTimingDelivery ) < 0 )
    {
        return -1;
    }

    derived().createROIndiText( m_indiP_dmTimingState, "dm_timing_state", "state" );
    m_indiP_dmTimingState.add( pcf::IndiElement( "clock" ) );
    m_indiP_dmTimingState.add( pcf::IndiElement( "receipt" ) );
    m_indiP_dmTimingState.add( pcf::IndiElement( "send" ) );
    m_indiP_dmTimingState.add( pcf::IndiElement( "deliveryClock" ) );
    m_indiP_dmTimingState["state"]   = m_dmTiming.enabled() ? "warmingUp" : "disabled";
    m_indiP_dmTimingState["clock"]   = "local: CLOCK_MONOTONIC; publicationInterval: stream writetime; seconds";
    m_indiP_dmTimingState["receipt"] = "first timestamp after successful semaphore wait return";
    m_indiP_dmTimingState["send"]    = "qualified backend submission return; not physical completion";
    const auto clock                 = m_dmTiming.summary().delivery.clock;
    m_indiP_dmTimingState["deliveryClock"] =
        clock == MagAOX::dmTiming::PublicationClock::tai
            ? "CLOCK_TAI"
            : ( clock == MagAOX::dmTiming::PublicationClock::realtime ? "CLOCK_REALTIME" : "disabled" );
    return derived().registerIndiPropertyReadOnly( m_indiP_dmTimingState );
}

template <class derivedT, typename realT>
dm<derivedT, realT>::~dm()
{
    for( auto &mi : m_channels )
    {
        if( mi != nullptr )
        {
            delete mi;
        }
    }
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::calibPath() const
{
    return m_calibPath;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::flatPath() const
{
    return m_flatPath;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::testPath() const
{
    return m_testPath;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::flatDefault() const
{
    return m_flatDefault;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::testDefault() const
{
    return m_testDefault;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::shmimFlat() const
{
    return m_shmimFlat;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::shmimTest() const
{
    return m_shmimTest;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::shmimSat() const
{
    return m_shmimSat;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::shmimSatPerc() const
{
    return m_shmimSatPerc;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::shmimShape() const
{
    return m_shmimShape;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::shmimDelta() const
{
    return m_shmimDelta;
}

template <class derivedT, typename realT>
uint32_t dm<derivedT, realT>::dmWidth() const
{
    return m_dmWidth;
}

template <class derivedT, typename realT>
uint32_t dm<derivedT, realT>::dmHeight() const
{
    return m_dmHeight;
}

template <class derivedT, typename realT>
uint8_t dm<derivedT, realT>::dmDataType() const
{
    return m_dmDataType;
}

template <class derivedT, typename realT>
float dm<derivedT, realT>::percThreshold() const
{
    return m_percThreshold;
}

template <class derivedT, typename realT>
float dm<derivedT, realT>::intervalSatThreshold() const
{
    return m_intervalSatThreshold;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::intervalSatCountThreshold() const
{
    return m_intervalSatCountThreshold;
}

template <class derivedT, typename realT>
const std::vector<std::string> &dm<derivedT, realT>::satTriggerDevice() const
{
    return m_satTriggerDevice;
}

template <class derivedT, typename realT>
const std::vector<std::string> &dm<derivedT, realT>::satTriggerProperty() const
{
    return m_satTriggerProperty;
}

template <class derivedT, typename realT>
const std::string &dm<derivedT, realT>::calibRelDir() const
{
    return m_calibRelDir;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::numChannels() const
{
    return m_numChannels;
}

template <class derivedT, typename realT>
const mx::improc::eigenImage<uint8_t> &dm<derivedT, realT>::instSatMap() const
{
    return m_instSatMap;
}

template <class derivedT, typename realT>
const mx::improc::eigenImage<uint16_t> &dm<derivedT, realT>::accumSatMap() const
{
    return m_accumSatMap;
}

template <class derivedT, typename realT>
const mx::improc::eigenImage<float> &dm<derivedT, realT>::satPercMap() const
{
    return m_satPercMap;
}

template <class derivedT, typename realT>
const std::vector<std::string> &dm<derivedT, realT>::deltaChannels() const
{
    return m_deltaChannels;
}

template <class derivedT, typename realT>
const std::vector<size_t> &dm<derivedT, realT>::notDeltas() const
{
    return m_notDeltas;
}

template <class derivedT, typename realT>
const mx::improc::eigenImage<float> &dm<derivedT, realT>::totalFlat() const
{
    return m_totalFlat;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setupConfig( mx::app::appConfigurator &config )
{
    config.add( "dm.readerReleaseStream",
                "",
                "dm.readerReleaseStream",
                argType::Required,
                "dm",
                "readerReleaseStream",
                false,
                "string",
                "Optional last-channel-read stream; SDK result is independent; empty disables." );

    config.add( "dm.completionStream",
                "",
                "dm.completionStream",
                argType::Required,
                "dm",
                "completionStream",
                false,
                "string",
                "Optional reader-completion stream with its own notification semaphores; empty disables." );

    config.add( "dm.calibPath",
                "",
                "dm.calibPath",
                argType::Required,
                "dm",
                "calibPath",
                false,
                "string",
                "The path to calibration files, relative to the MagAO-X calibration path." );

    config.add( "dm.flatPath",
                "",
                "dm.flatPath",
                argType::Required,
                "dm",
                "flatPath",
                false,
                "string",
                "The path to flat files.  Default is the calibration path." );

    config.add( "dm.flatDefault",
                "",
                "dm.flatDefault",
                argType::Required,
                "dm",
                "flatDefault",
                false,
                "string",
                "The default flat file (path and extension are not required)." );

    config.add( "dm.testPath",
                "",
                "dm.testPath",
                argType::Required,
                "dm",
                "testPath",
                false,
                "string",
                "The path to test files.  Default is the calibration path plus /tests." );

    config.add( "dm.testDefault",
                "",
                "dm.testDefault",
                argType::Required,
                "dm",
                "testDefault",
                false,
                "string",
                "The default test file (path and extension are not required)." );

    config.add( "dm.actMaskPath",
                "",
                "dm.actMaskPath",
                argType::Required,
                "dm",
                "actMaskPath",
                false,
                "string",
                "The path to the actuator mask for this DM, relative to the calib path." );

    // Overriding the shmimMonitor setup so that these all go in the dm section
    // Otherwise, would call shmimMonitor<dm<derivedT,realT>>::setupConfig();
    ///\todo shmimMonitor now has configSection so this isn't necessary.
    config.add( "dm.latencyTime",
                "",
                "dm.latencyTime",
                argType::Required,
                "dm",
                "latencyTime",
                false,
                "double",
                "Trailing DM timing window in seconds; zero disables." );
    config.add( "dm.latencySize",
                "",
                "dm.latencySize",
                argType::Required,
                "dm",
                "latencySize",
                false,
                "size_t",
                "Maximum queued and retained DM timing events; zero disables." );

    config.add( "dm.deliveryClock",
                "",
                "dm.deliveryClock",
                argType::Required,
                "dm",
                "deliveryClock",
                false,
                "string",
                "Explicit input-producer clock for delivery diagnostics: none (default), isio, realtime or tai." );
    config.add( "dm.deliveryBracketUs",
                "",
                "dm.deliveryBracketUs",
                argType::Required,
                "dm",
                "deliveryBracketUs",
                false,
                "double",
                "Maximum accepted delivery clock-capture bracket in microseconds; default 1." );

    config.add( "dm.threadPrio",
                "",
                "dm.threadPrio",
                argType::Required,
                "dm",
                "threadPrio",
                false,
                "int",
                "The real-time priority of the dm control thread." );

    config.add( "dm.cpuset",
                "",
                "dm.cpuset",
                argType::Required,
                "dm",
                "cpuset",
                false,
                "int",
                "The cpuset for the dm control thread." );

    config.add( "dm.shmimName",
                "",
                "dm.shmimName",
                argType::Required,
                "dm",
                "shmimName",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to monitor for DM comands. Will be used as "
                "/tmp/<shmimName>.im.shm." );

    // end of shmimmonitor overrides

    config.add( "dm.shmimFlat",
                "",
                "dm.shmimFlat",
                argType::Required,
                "dm",
                "shmimFlat",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to write the flat command to.  Default is shmimName "
                "with 00 apended (i.e. dm00disp -> dm00disp00). " );

    config.add( "dm.shmimTest",
                "",
                "dm.shmimTest",
                argType::Required,
                "dm",
                "shmimTest",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to write the test command to.  Default is shmimName "
                "with 01 apended (i.e. dm00disp -> dm00disp01). " );

    config.add( "dm.shmimSat",
                "",
                "dm.shmimSat",
                argType::Required,
                "dm",
                "shmimSat",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to write the saturation map to.  Default is "
                "shmimName with SA apended (i.e. dm00disp -> dm00dispSA).  This is created." );

    config.add( "dm.shmimSatPerc",
                "",
                "dm.shmimSatPerc",
                argType::Required,
                "dm",
                "shmimSatPerc",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to write the saturation percentage map to.  Default "
                "is shmimName with SP apended (i.e. dm00disp -> dm00dispSP).  This is created." );

    config.add( "dm.satAvgInt",
                "",
                "dm.satAvgInt",
                argType::Required,
                "dm",
                "satAvgInt",
                false,
                "int",
                "The interval in milliseconds over which saturation "
                "is accumulated before updating.  Default is 100 ms." );

    config.add( "dm.satThreadPrio",
                "",
                "dm.satThreadPrio",
                argType::Required,
                "dm",
                "satThreadPrio",
                false,
                "int",
                "The priority for the saturation thread. "
                "Usually ok to be 0." );

    config.add( "dm.shmimShape",
                "",
                "dm.shmimShape",
                argType::Required,
                "dm",
                "shmimShape",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to write the desaturated shape to.  Default is "
                "shmimName with _shape apended (i.e. dm00disp -> dm00disp_shape).  This is created." );

    config.add( "dm.shmimDelta",
                "",
                "dm.shmimDelta",
                argType::Required,
                "dm",
                "shmimDelta",
                false,
                "string",
                "The name of the ImageStreamIO shared memory image to write the "
                "desaturated delta-shape to.  Default is "
                "shmimName with _delta apended (i.e. dm00disp -> dm00disp_delta).  This is created." );

    config.add( "dm.deltaChannels",
                "",
                "dm.deltaChannels",
                argType::Required,
                "dm",
                "deltaChannels",
                false,
                "vector<string>",
                "The names of the DM channels which are delta commands to be excluded from the total flat." );

    config.add( "dm.width",
                "",
                "dm.width",
                argType::Required,
                "dm",
                "width",
                false,
                "string",
                "The width of the DM in actuators." );

    config.add( "dm.height",
                "",
                "dm.height",
                argType::Required,
                "dm",
                "height",
                false,
                "string",
                "The height of the DM in actuators." );

    config.add( "dm.percThreshold",
                "",
                "dm.percThreshold",
                argType::Required,
                "dm",
                "percThreshold",
                false,
                "float",
                "Threshold on percentage of frames an actuator is saturated over an interval.  Default is 0.98." );

    config.add( "dm.intervalSatThreshold",
                "",
                "dm.intervalSatThreshold",
                argType::Required,
                "dm",
                "intervalSatThreshold",
                false,
                "float",
                "Threshold on percentage of actuators which exceed percThreshold in an interval.  Default is 0.5." );

    config.add( "dm.intervalSatCountThreshold",
                "",
                "dm.intervalSatCountThreshold",
                argType::Required,
                "dm",
                "intervalSatCountThreshold",
                false,
                "float",
                "Threshold on number of consecutive intervals the intervalSatThreshold is exceeded.  Default is 10." );

    config.add( "dm.satTriggerDevice",
                "",
                "dm.satTriggerDevice",
                argType::Required,
                "dm",
                "satTriggerDevice",
                false,
                "vector<string>",
                "Device(s) with a toggle switch to toggle on saturation trigger." );

    config.add( "dm.satTriggerProperty",
                "",
                "dm.satTriggerProperty",
                argType::Required,
                "dm",
                "satTriggerProperty",
                false,
                "vector<string>",
                "Property with a toggle switch to toggle on saturation trigger, one per entry in satTriggerDevice." );

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::loadConfig( mx::app::appConfigurator &config )
{

    m_calibPath = derived().m_calibDir + "/" + m_calibRelDir;
    config( m_calibPath, "dm.calibPath" );

    // setup flats
    m_flatPath = m_calibPath + "/flats";
    config( m_flatPath, "dm.flatPath" );

    config( m_flatDefault, "dm.flatDefault" );

    if( m_flatDefault != "" )
    {
        m_flatDefault = mx::ioutils::pathStem( m_flatDefault ); // strip off path and extension if provided.
        m_flatCurrent = "default";
    }

    // setup tests
    m_testPath = m_calibPath + "/tests";
    config( m_testPath, "dm.testPath" );

    config( m_testDefault, "dm.testDefault" );

    if( m_testDefault != "" )
    {
        m_testDefault = mx::ioutils::pathStem( m_testDefault ); // strip off path and extension if provided.
        m_testCurrent = "default";
    }

    config( m_actMaskPath, "dm.actMaskPath" );

    // Overriding the shmimMonitor setup so that these all go in the dm section
    // Otherwise, would call shmimMonitor<dm<derivedT,realT>>::loadConfig(config);
    config( derived().m_smThreadPrio, "dm.threadPrio" );
    config( derived().m_smCpuset, "dm.cpuset" );

    config( derived().m_shmimName, "dm.shmimName" );
    config( m_completionStreamName, "dm.completionStream" );
    config( m_readerReleaseStreamName, "dm.readerReleaseStream" );

    if( !m_completionStreamName.empty() && ( m_completionStreamName == derived().m_shmimName ||
                                             m_completionStreamName.find_first_of( "/\\" ) != std::string::npos ) )
    {
        derivedT::template log<software_error>( { "invalid dm.completionStream" } );
        return -1;
    }

    config( m_timingWindow, "dm.latencyTime" );
    config( m_timingCapacity, "dm.latencySize" );
    config( m_timingDeliveryClockName, "dm.deliveryClock" );
    config( m_timingDeliveryBracketUs, "dm.deliveryBracketUs" );

    if( !parseDeliveryClock( m_timingDeliveryClockName, m_timingDeliveryClock ) ||
        !mx::math::isFinite( m_timingDeliveryBracketUs ) || m_timingDeliveryBracketUs < 0.001 ||
        m_timingDeliveryBracketUs > 1000000 )
    {
        derivedT::template log<software_error>( { "invalid dm.deliveryClock or dm.deliveryBracketUs" } );
        return -1;
    }

    if( !mx::math::isFinite( m_timingWindow ) || m_timingWindow < 0 || m_timingWindow >= 9e9 )
    {
        derivedT::template log<software_error>( { "invalid dm.latencyTime" } );
        return -1;
    }

    derived().m_getExistingFirst = true;
    // end of shmimmonitor overrides

    if( derived().m_shmimName != "" )
    {
        m_shmimFlat = derived().m_shmimName + "00";
        config( m_shmimFlat, "dm.shmimFlat" );

        m_shmimTest = derived().m_shmimName + "02";
        config( m_shmimTest, "dm.shmimTest" );

        m_shmimSat = derived().m_shmimName + "ST";
        config( m_shmimSat, "dm.shmimSat" );

        m_shmimSatPerc = derived().m_shmimName + "SP";
        config( m_shmimSatPerc, "dm.shmimSatPerc" );

        config( m_satAvgInt, "dm.satAvgInt" );

        config( m_satThreadPrio, "dm.satSatThreadPrio" );

        m_shmimShape = derived().m_shmimName + "_shape";
        config( m_shmimShape, "dm.shmimShape" );

        m_shmimDelta = derived().m_shmimName + "_delta";
        config( m_shmimDelta, "dm.shmimDelta" );

        m_shmimDiff = derived().m_shmimName + "_diff";
        config( m_shmimDiff, "dm.shmimDiff" );

        config( m_deltaChannels, "dm.deltaChannels" );
    }
    else
    {
        // Avoid unused error
        config.isSet( "dm.shmimFlat" );
        config.isSet( "dm.shmimTest" );
        config.isSet( "dm.shmimSat" );
        config.isSet( "dm.shmimSatPerc" );
        config.isSet( "dm.satAvgInt" );
        config.isSet( "dm.shmimShape" );
        config.isSet( "dm.shmimDelta" );
        config.isSet( "dm.deltaChannels" );
    }

    if( !m_readerReleaseStreamName.empty() )
    {
        const std::vector<std::string> reserved{ derived().m_shmimName,
                                                 m_completionStreamName,
                                                 m_shmimFlat,
                                                 m_shmimTest,
                                                 m_shmimSat,
                                                 m_shmimSatPerc,
                                                 m_shmimShape,
                                                 m_shmimDelta,
                                                 m_shmimDiff,
                                                 derived().m_shmimName + "_actmask" };

        if( m_readerReleaseStreamName.find_first_not_of(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" ) != std::string::npos ||
            std::find( reserved.begin(), reserved.end(), m_readerReleaseStreamName ) != reserved.end() ||
            std::find( m_deltaChannels.begin(), m_deltaChannels.end(), m_readerReleaseStreamName ) !=
                m_deltaChannels.end() ||
            m_readerReleaseStreamName.starts_with( derived().m_shmimName ) )
        {
            derivedT::template log<software_error>( { "invalid or conflicting dm.readerReleaseStream" } );
            return -1;
        }
    }

    config( m_dmWidth, "dm.width" );
    config( m_dmHeight, "dm.height" );

    config( m_percThreshold, "dm.percThreshold" );
    config( m_intervalSatThreshold, "dm.intervalSatThreshold" );
    config( m_intervalSatCountThreshold, "dm.intervalSatCountThreshold" );
    config( m_satTriggerDevice, "dm.satTriggerDevice" );
    config( m_satTriggerProperty, "dm.satTriggerProperty" );

    if( m_dmWidth > 0 && m_dmHeight > 0 )
    {
        try
        {
            m_actMask.create( derived().m_shmimName + "_actmask", m_dmWidth, m_dmHeight );
        }
        catch( const std::exception &e )
        {
            derivedT::template log<text_log>( std::format( "exception caught creating actuator mask: "
                                                           "{}: {}",
                                                           derived().m_shmimName + "_actmask",
                                                           e.what(),
                                                           logPrio::LOG_ERROR ) );
            return -1;
        }

        if( m_actMaskPath != "" )
        {
            mx::improc::eigenImage<realT> actMask;

            mx::fits::fitsFile<realT> ff;

            mx::error_t errc = ff.read( actMask, m_calibPath + '/' + m_actMaskPath );

            if( errc != mx::error_t::noerror )
            {
                derivedT::template log<text_log>( std::format( "error reading actuator mask file {}: "
                                                               "{} ({})",
                                                               m_calibPath + '/' + m_actMaskPath,
                                                               mx::errorMessage( errc ),
                                                               mx::errorName( errc ) ),
                                                  logPrio::LOG_ERROR );
                return -1;
            }

            if( actMask.rows() != m_dmWidth || actMask.cols() != m_dmHeight )
            {
                derivedT::template log<text_log>( std::format( "actuaor mask {}x{} is not same size as flag {}x{}",
                                                               actMask.rows(),
                                                               actMask.cols(),
                                                               m_dmWidth,
                                                               m_dmHeight ),
                                                  logPrio::LOG_ERROR );

                return -1;
            }

            m_actMask = actMask;
        }
        else
        {
            m_actMask().setConstant( 1.0 );
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::appStartup()
{
    try
    {
        m_dmTiming.configure( m_timingWindow,
                              m_timingCapacity,
                              0,
                              m_timingDeliveryClock,
                              static_cast<int64_t>( m_timingDeliveryBracketUs * 1000 ) );
    }
    catch( const std::exception &error )
    {
        derivedT::template log<software_error>( { std::string( "DM timing allocation: " ) + error.what() } );
        return -1;
    }

    if( setupDMTimingINDI() < 0 )
    {
        return -1;
    }

    if( m_dmCompletion.open( m_completionStreamName ) < 0 )
    {
        derivedT::template log<software_error>( { errno, "could not open DM completion stream" } );
        return -1;
    }

    if( m_dmReaderRelease.open( m_readerReleaseStreamName, MagAOX::dmCompletion::Kind::readers ) < 0 )
    {
        derivedT::template log<software_error>( { errno, "could not open DM reader-release stream" } );
        return -1;
    }

    if( setupDMCompletionINDI() < 0 )
    {
        return -1;
    }

    if( m_dmDataType == 0 )
    {
        derivedT::template log<software_error>( { "unsupported DM data type" } );
        return -1;
    }

    //-----------------
    // Get the flats
    checkFlats();

    // Register the test shmim INDI property
    m_indiP_flatShmim = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_flatShmim.setDevice( derived().configName() );
    m_indiP_flatShmim.setName( "flat_shmim" );
    m_indiP_flatShmim.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_flatShmim.setState( pcf::IndiProperty::Idle );
    m_indiP_flatShmim.add( pcf::IndiElement( "channel" ) );
    m_indiP_flatShmim["channel"] = m_shmimFlat;

    if( derived().registerIndiPropertyReadOnly( m_indiP_flatShmim ) < 0 )
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( { "" } );
#endif
        return -1;
    }

    // Register the setFlat INDI property
    derived().createStandardIndiToggleSw( m_indiP_setFlat, "flat_set" );

    if( derived().registerIndiPropertyNew( m_indiP_setFlat, st_newCallBack_setFlat ) < 0 )
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( { "" } );
#endif
        return -1;
    }

    //-----------------
    // Get the tests
    checkTests();

    // Register the test shmim INDI property
    m_indiP_testShmim = pcf::IndiProperty( pcf::IndiProperty::Text );
    m_indiP_testShmim.setDevice( derived().configName() );
    m_indiP_testShmim.setName( "test_shmim" );
    m_indiP_testShmim.setPerm( pcf::IndiProperty::ReadOnly );
    m_indiP_testShmim.setState( pcf::IndiProperty::Idle );
    m_indiP_testShmim.add( pcf::IndiElement( "channel" ) );
    m_indiP_testShmim["channel"] = m_shmimTest;
    derived().createStandardIndiToggleSw( m_indiP_setTest, "test_shmim" );

    if( derived().registerIndiPropertyReadOnly( m_indiP_testShmim ) < 0 )
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( { "" } );
#endif
        return -1;
    }

    // Register the setTest INDI property
    derived().createStandardIndiToggleSw( m_indiP_setTest, "test_set" );

    if( derived().registerIndiPropertyNew( m_indiP_setTest, st_newCallBack_setTest ) < 0 )
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( { "" } );
#endif
        return -1;
    }

    // Register the init INDI property
    derived().createStandardIndiRequestSw( m_indiP_init, "initDM" );

    if( derived().registerIndiPropertyNew( m_indiP_init, st_newCallBack_init ) < 0 )
    {
        // clang-format off
        #ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( {""} );
        #endif
        // clang-format on

        return -1;
    }

    // Register the zero INDI property
    derived().createStandardIndiRequestSw( m_indiP_zero, "zeroDM" );

    if( derived().registerIndiPropertyNew( m_indiP_zero, st_newCallBack_zero ) < 0 )
    {
        // clang-format off
        #ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( {""} );
        #endif
        // clang-format on

        return -1;
    }

    // Register the release INDI property
    derived().createStandardIndiRequestSw( m_indiP_release, "releaseDM" );

    if( derived().registerIndiPropertyNew( m_indiP_release, st_newCallBack_release ) < 0 )
    {
        return derivedT::template log<software_error, -1>( { "" } );
    }

    derived().createStandardIndiRequestSw( m_indiP_zeroAll, "zeroAll" );

    if( derived().registerIndiPropertyNew( m_indiP_zeroAll, st_newCallBack_zeroAll ) < 0 )
    {
#ifndef DM_TEST_NOLOG
        derivedT::template log<software_error>( { "" } );
#endif
        return -1;
    }

    if( m_flatDefault != "" )
    {
        loadFlat( "default" );
    }

    if( m_testDefault != "" )
    {
        loadTest( "default" );
    }

    if( sem_init( &m_satSemaphore, 0, 0 ) < 0 )
    {
        return derivedT::template log<software_critical, -1>( { errno, 0, "Initializing sat semaphore" } );
    }

    if( derived().threadStart( m_satThread,
                               m_satThreadInit,
                               m_satThreadID,
                               m_satThreadProp,
                               m_satThreadPrio,
                               "",
                               "saturation",
                               this,
                               satThreadStart ) < 0 )
    {
        derivedT::template log<software_error, -1>( { "" } );
        return -1;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::appLogic()
{
    // do a join check to see if other threads have exited.
    if( pthread_tryjoin_np( m_satThread.native_handle(), 0 ) == 0 )
    {
        derivedT::template log<software_error>( { "saturation thread has exited" } );

        return -1;
    }

    checkFlats();

    checkTests();

    if( m_intervalSatTrip )
    {
        intervalSatTrip();
        m_intervalSatTrip = false;
    }

    updateDMTimings();
    recordDMTimings();

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::appShutdown()
{
    if( m_dmCompletion.enabled() || m_dmReaderRelease.enabled() )
    {
        // Existing apps call DM shutdown before monitor shutdown; join its sole publisher before closing the stream.
        static_cast<dev::shmimMonitor<derivedT> &>( derived() ).appShutdown();
        m_dmCompletion.close();
        m_dmReaderRelease.close();
    }

    if( m_satThread.joinable() )
    {
        pthread_kill( m_satThread.native_handle(), SIGUSR1 );
        try
        {
            m_satThread.join(); // this will throw if it was already joined
        }
        catch( ... )
        {
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::onPowerOff()
{
    baseReleaseDM();

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::whilePowerOff()
{
    checkFlats();
    checkTests();
    updateDMTimings();

    if( recordDMTimings() < 0 )
    {
        return -1;
    }

    return updateINDI();
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::findDMChannels()
{
    std::string milkShmimDir = mx::sys::getEnv( "MILK_SHM_DIR" );

    if( milkShmimDir == "" )
    {
        milkShmimDir = "/milk/shm";
    }

    std::vector<std::string> dmlist;
    mx::error_t errc = mx::ioutils::getFileNames( dmlist, milkShmimDir, derived().m_shmimName, ".im", ".shm" );

    mx_error_check_rv( errc, -1 );

    if( dmlist.size() == 0 )
    {
        derivedT::template log<software_error>( { "no dm channels found for " + derived().m_shmimName } );

        return -1;
    }

    m_numChannels = -1;

    for( size_t n = 0; n < dmlist.size(); ++n )
    {
        char nstr[16];
        snprintf( nstr, sizeof( nstr ), "%02d.im.shm", (int)n );
        std::string tgt = derived().m_shmimName;
        tgt += nstr;

        for( size_t m = 0; m < dmlist.size(); ++m )
        {
            if( dmlist[m].find( tgt ) != std::string::npos )
            {
                if( (int)n > m_numChannels )
                {
                    m_numChannels = n;
                }
            }
        }
    }

    ++m_numChannels;

    derivedT::template log<text_log>(
        { std::format( "Found {} chanels for {} ", m_numChannels, derived().m_shmimName ) } );

    m_channels.resize( m_numChannels, nullptr );

    m_notDeltas.clear();
    m_deltas.clear();

    for( size_t n = 0; n < m_channels.size(); ++n )
    {
        std::string sname = std::format( "{}{:02}", derived().m_shmimName, n );

        try
        {
            m_channels[n] = new mx::improc::milkImage<realT>( sname ); // this opens the channel stream
        }
        catch( const std::exception &e )
        {
            derivedT::template log<software_error>( { "exception opening " + sname + ": " + e.what() } );
        }

        std::cerr << "looking for " << sname << '\n';
        auto res = std::find( m_deltaChannels.begin(), m_deltaChannels.end(), sname );

        if( res == m_deltaChannels.end() )
        {
            std::cerr << "  not a delta\n";
            m_notDeltas.push_back( n );
        }
        else
        {
            std::cerr << "  is a delta\n";
            m_deltas.push_back( n );
        }
    }

    std::cerr << "not deltas: ";

    for( size_t n = 0; n < m_notDeltas.size(); ++n )
    {
        std::cerr << m_notDeltas[n] << ' ';
    }

    std::cerr << '\n';

    std::cerr << "deltas: ";

    for( size_t n = 0; n < m_deltas.size(); ++n )
    {
        std::cerr << m_deltas[n] << ' ';
    }

    std::cerr << '\n';

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::allocate( const dev::shmimT &sp )
{
    std::unique_lock<std::mutex> pipelineLock( m_pipelineSatMutex, std::defer_lock );

    if( m_pipelineSatEnabled.load( std::memory_order_acquire ) )
    {
        pipelineLock.lock();
        m_pipelineSaturation.configure( size_t( m_dmWidth ) * m_dmHeight );
        m_pipelineSatWindow.resize( m_dmWidth, m_dmHeight );
        m_pipelineSatWindow.setZero();
        ++m_pipelineSatGeneration;
    }

    static_cast<void>( sp ); // be unused

    int err = 0;

    if( derived().m_width != m_dmWidth )
    {
        derivedT::template log<software_critical>( { "shmim width does not match configured DM width" } );
        ++err;
    }

    if( derived().m_height != m_dmHeight )
    {
        derivedT::template log<software_critical>( { "shmim height does not match configured DM height" } );
        ++err;
    }

    if( derived().m_dataType != m_dmDataType )
    {
        derivedT::template log<software_critical>( { "shmim data type does not match configured DM data type" } );
        ++err;
    }

    if( err )
    {
        return -1;
    }

    m_instSatMap.resize( m_dmWidth, m_dmHeight );
    m_instSatMap.setZero();

    m_accumSatMap.resize( m_dmWidth, m_dmHeight );
    m_accumSatMap.setZero();

    m_satPercMap.resize( m_dmWidth, m_dmHeight );
    m_satPercMap.setZero();

    if( findDMChannels() < 0 )
    {
        derivedT::template log<software_critical>( { "error finding DM channels" } );

        return -1;
    }

    try
    {
        m_outputShape.create( m_shmimShape, m_dmWidth, m_dmHeight );
        m_outputShape().setZero();
    }
    catch( const std::exception &e )
    {
        return derivedT::template log<software_error, -1>(
            { std::string( "creating output shape shmim: " ) + e.what() } );
    }

    try
    {
        m_outputDelta.create( m_shmimDelta, m_dmWidth, m_dmHeight );
        m_outputDelta().setZero();
    }
    catch( const std::exception &e )
    {
        return derivedT::template log<software_error, -1>(
            { std::string( "creating output delta shmim: " ) + e.what() } );
    }

    try
    {
        m_outputDiff.create( m_shmimDiff, m_dmWidth, m_dmHeight );
        m_outputDiff().setZero();
    }
    catch( const std::exception &e )
    {
        return derivedT::template log<software_error, -1>(
            { std::string( "creating output diff shmim: " ) + e.what() } );
    }

    m_totalFlat.resize( m_dmWidth, m_dmHeight );
    m_totalFlat.setZero();

    m_totalDelta.resize( m_dmWidth, m_dmHeight );
    m_totalDelta.setZero();

    m_stagedDelta.resize( m_dmWidth, m_dmHeight );
    m_stagedDiff.resize( m_dmWidth, m_dmHeight );
    m_deltaStaged = false;

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::processImage( void *curr_src, const dev::shmimT &sp )
{
    static_cast<void>( sp );

    if( !derived().admitCommand() )
    {
        m_dmTiming.outcome( MagAOX::dmTiming::Outcome::rejected );
        m_dmTiming.finishEvent();
        m_completionReceipt = false;
        m_dmCompletion.reset();
        m_dmReaderRelease.reset();

        return 0;
    }

    CommandRetirement retirement( *this );
    m_completionOutcome       = MagAOX::dmTiming::Outcome::unknown;
    m_completionBackendResult = MagAOX::dmCompletion::unavailableResult;
    m_deltaStaged             = false;
    m_dmCompletion.begin( &derived().m_imageStream, m_completionReceipt );
    m_dmReaderRelease.begin( &derived().m_imageStream, m_completionReceipt );
    m_completionReceipt = false;

    if( !m_dmTiming.active() )
    {
        m_dmTiming.beginEvent( false );
    }

    dmTimingMark( MagAOX::dmTiming::Marker::entry );

    if( m_dmTiming.active() && derived().m_imageStream.md != nullptr && derived().m_imageStream.md->write == 0 )
    {
        const auto *metadata = derived().m_imageStream.md;
        uint64_t    counter  = metadata->cnt0;
        const auto  stamp    = metadata->writetime;
        std::atomic_thread_fence( std::memory_order_acquire );

        // Best-effort metadata observation; this does not establish atomic token/pixel identity.
        if( metadata->write == 0 && counter == metadata->cnt0 )
        {
            m_dmTiming.inputCounter( counter );

            if( stamp.tv_sec >= 0 && stamp.tv_nsec >= 0 && stamp.tv_nsec < 1000000000 &&
                stamp.tv_sec <= ( std::numeric_limits<int64_t>::max() - stamp.tv_nsec ) / 1000000000 )
            {
                m_dmTiming.inputPublication( static_cast<int64_t>( stamp.tv_sec ) * 1000000000 + stamp.tv_nsec );
            }
        }
    }

    int  rv       = -1;
    bool deferred = false;

    try
    {
        rv       = derived().commandDM( curr_src );
        deferred = derived().deferredCommand();

        if( rv >= 0 )
        {
            if( !m_deltaChannels.empty() )
            {
                dmTimingMark( MagAOX::dmTiming::Marker::deltaStart );

                if( deferred )
                {
                    dmTimingMark( MagAOX::dmTiming::Marker::stagingStart );
                    rv = stageDelta();
                    dmTimingMark( MagAOX::dmTiming::Marker::stagingEnd );
                }
                else
                {
                    rv = makeDelta();
                    dmTimingMark( MagAOX::dmTiming::Marker::deltaEnd );
                }
            }
            else if( !deferred )
            {
                // Preserve the synchronous processing-through-delta endpoint without delta work.
                dmTimingMark( MagAOX::dmTiming::Marker::deltaEnd );
            }
        }
    }
    catch( const std::exception &error )
    {
        rv = -1;
        derivedT::template log<software_critical>( { std::string( "DM command processing: " ) + error.what() } );
    }
    catch( ... )
    {
        rv = -1;
        derivedT::template log<software_critical>( { "unknown exception processing DM command" } );
    }

    bool coherentIdentity = true;

    if( m_dmCompletion.enabled() && m_dmReaderRelease.enabled() )
    {
        const auto &complete = m_dmCompletion.attempt();
        const auto &readers  = m_dmReaderRelease.attempt();
        coherentIdentity = complete.inputInode == readers.inputInode && complete.inputCounter == readers.inputCounter &&
                           complete.publicationSec == readers.publicationSec &&
                           complete.publicationNsec == readers.publicationNsec;
    }

    m_dmReaderRelease.release( &derived().m_imageStream, rv, coherentIdentity );

    if( rv >= 0 )
    {
        if( m_dmReaderRelease.enabled() )
        {
            m_dmTiming.markAt( MagAOX::dmTiming::Marker::readRelease, m_dmReaderRelease.attempt().finishedNs );
        }
        else
        {
            dmTimingMark( MagAOX::dmTiming::Marker::readRelease );
        }
    }

    // Finalization drains the backend even after command preparation or private staging failed.
    const int processingResult = rv;
    const int finalResult      = derived().finalizeCommand( processingResult );
    retirement.m_finalized     = true;
    rv                         = processingResult < 0 ? processingResult : finalResult;

    // Freeze actual submission evidence before a later publication/post error changes callback status.
    m_dmReaderRelease.submission( m_completionOutcome, m_completionBackendResult );

    if( rv >= 0 && deferred )
    {
        try
        {
            dmTimingMark( MagAOX::dmTiming::Marker::publicationStart );
            rv = publishDelta();
            dmTimingMark( MagAOX::dmTiming::Marker::publicationEnd );
        }
        catch( const std::exception &error )
        {
            rv = -1;
            dmTimingOutcome( MagAOX::dmTiming::Outcome::failed );
            derivedT::template log<software_critical>( { std::string( "DM delta publication: " ) + error.what() } );
        }
        catch( ... )
        {
            rv = -1;
            dmTimingOutcome( MagAOX::dmTiming::Outcome::failed );
            derivedT::template log<software_critical>( { "unknown exception publishing DM delta" } );
        }

        dmTimingMark( MagAOX::dmTiming::Marker::deltaEnd );
    }

    m_deltaStaged = false;

    if( rv >= 0 )
    {
        dmTimingMark( MagAOX::dmTiming::Marker::postStart );
        const int postResult = sem_post( &m_satSemaphore );
        dmTimingMark( MagAOX::dmTiming::Marker::postEnd );

        if( postResult < 0 )
        {
            rv = -1;
            dmTimingOutcome( MagAOX::dmTiming::Outcome::failed );
            derivedT::template log<software_critical>( { errno, 0, "Error posting to semaphore" } );
        }
    }

    if( rv < 0 && m_completionOutcome != MagAOX::dmTiming::Outcome::partial )
    {
        dmTimingOutcome( MagAOX::dmTiming::Outcome::failed );
        derivedT::template log<software_critical>( { errno, rv, "DM command callback failed" } );
    }

    dmTimingMark( MagAOX::dmTiming::Marker::finish );
    m_dmTiming.finishEvent();

    m_dmCompletion.finish( &derived().m_imageStream, rv, m_completionOutcome );

    return rv;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::baseInitDM()
{
    if( derived().state() != stateCodes::NOTHOMED )
    {
        derivedT::template log<software_error>( { errno, "DM is not ready to be initialized" } );
        derived().state( stateCodes::ERROR );
        return -1;
    }

    derived().state( stateCodes::HOMING );

    int rv;

    if( ( rv = derived().initDM() ) < 0 )
    {
        derivedT::template log<software_critical>( { errno, rv, "Error from initDM" } );
        derived().state( stateCodes::ERROR );
        return rv;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::baseReleaseDM()
{
    if( derived().state() != stateCodes::POWEROFF )
    {
        derived().state( stateCodes::NOTHOMED );
    }

    int rv;

    if( ( rv = derived().releaseDM() ) < 0 )
    {
        derivedT::template log<software_critical>( { errno, rv, "Error from releaseDM" } );
        derived().state( stateCodes::ERROR );
        return rv;
    }

    if( ( rv = zeroAll( true ) ) < 0 )
    {
        derivedT::template log<software_error>( { errno, rv, "Error from zeroAll" } );
        derived().state( stateCodes::ERROR );
        return rv;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::checkFlats()
{
    std::vector<std::string> tfs;

    bool gfn_logged = false;

    mx::error_t errc = mx::ioutils::getFileNames( tfs, m_flatPath, "", "", ".fits" );

    if( errc != mx::error_t::noerror )
    {
        if( !gfn_logged )
        {
            derivedT::template log<software_error>(
                { std::format( "error getting flat files: {}", mx::errorMessage( errc ) ) } );
        }

        gfn_logged = true;
        return -1;
    }

    gfn_logged = false;

    // First remove default, b/c we always add it and don't want to include it in timestamp selected ones
    for( size_t n = 0; n < tfs.size(); ++n )
    {
        if( mx::ioutils::pathStem( tfs[n] ) == "default" )
        {
            tfs.erase( tfs.begin() + n );
            --n;
        }
    }

    unsigned m_nFlatFiles = 5;

    // Here we keep only the m_nFlatFiles most recent files
    if( tfs.size() >= m_nFlatFiles )
    {
        std::vector<std::filesystem::file_time_type> wtimes( tfs.size() );

        for( size_t n = 0; n < wtimes.size(); ++n )
        {
            wtimes[n] = std::filesystem::last_write_time( tfs[n] );
        }

        std::sort( wtimes.begin(), wtimes.end() );

        std::filesystem::file_time_type tn = wtimes[wtimes.size() - m_nFlatFiles];

        for( size_t n = 0; n < tfs.size(); ++n )
        {
            std::filesystem::file_time_type lmt = std::filesystem::last_write_time( tfs[n] );

            if( lmt < tn )
            {
                tfs.erase( tfs.begin() + n );
                --n;
            }
        }
    }

    for( auto it = m_flatCommands.begin(); it != m_flatCommands.end(); ++it )
    {
        it->second = "";
    }

    bool changed = false;

    for( size_t n = 0; n < tfs.size(); ++n )
    {
        auto ir =
            m_flatCommands.insert( std::pair<std::string, std::string>( mx::ioutils::pathStem( tfs[n] ), tfs[n] ) );

        if( ir.second == true )
        {
            changed = true;
        }
        else
        {
            ir.first->second = tfs[n];
        }
    }

    for( auto it = m_flatCommands.begin(); it != m_flatCommands.end(); ++it )
    {
        if( it->second == "" )
        {
            changed = true;
            // Erase the current iterator safely, even if the first one.
            auto itdel = it;
            ++it;
            m_flatCommands.erase( itdel );
            --it;
        };
    }

    if( changed )
    {
        if( derived().m_indiDriver )
        {
            derived().m_indiDriver->sendDelProperty( m_indiP_flats );
            derived().m_indiNewCallBacks.erase( m_indiP_flats.createUniqueKey() );
        }

        m_indiP_flats = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_flats.setDevice( derived().configName() );
        m_indiP_flats.setName( "flat" );
        m_indiP_flats.setPerm( pcf::IndiProperty::ReadWrite );
        m_indiP_flats.setState( pcf::IndiProperty::Idle );
        m_indiP_flats.setRule( pcf::IndiProperty::OneOfMany );

        // Add the toggle element initialized to Off
        for( auto it = m_flatCommands.begin(); it != m_flatCommands.end(); ++it )
        {
            if( it->first == m_flatCurrent || m_flatCurrent == "" )
            {
                m_indiP_flats.add( pcf::IndiElement( it->first, pcf::IndiElement::On ) );
                m_flatCurrent = it->first; // handles the case m_flatCurrent == "" b/c it was not set in config
            }
            else
            {
                m_indiP_flats.add( pcf::IndiElement( it->first, pcf::IndiElement::Off ) );
            }
        }

        if( m_flatDefault != "" )
        {
            if( m_flatCurrent == "default" )
            {
                m_indiP_flats.add( pcf::IndiElement( "default", pcf::IndiElement::On ) );
            }
            else
            {
                m_indiP_flats.add( pcf::IndiElement( "default", pcf::IndiElement::Off ) );
            }
        }

        if( derived().registerIndiPropertyNew( m_indiP_flats, st_newCallBack_flats ) < 0 )
        {
            // clang-format off
            #ifndef DM_TEST_NOLOG
            derivedT::template log<software_error>( {""} );
            #endif
            // clang-format on

            return -1;
        }

        if( derived().m_indiDriver )
        {
            derived().m_indiDriver->sendDefProperty( m_indiP_flats );
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::loadFlat( const std::string &intarget )
{
    std::string target = intarget;

    std::string targetPath;

    if( target == "default" )
    {
        target     = m_flatDefault;
        targetPath = m_flatPath + "/" + m_flatDefault + ".fits";
    }
    else
    {
        try
        {
            targetPath = m_flatCommands.at( target );
        }
        catch( ... )
        {
            derivedT::template log<text_log>( "flat file " + target + " not found", logPrio::LOG_ERROR );
            return -1;
        }
    }

    m_flatLoaded = false;

    // load into memory.
    mx::fits::fitsFile<realT> ff;

    mx::error_t errc = ff.read( m_flatCommand, targetPath );

    if( errc != mx::error_t::noerror )
    {
        derivedT::template log<text_log>( std::format( "error reading flat file {}: "
                                                       "{} ({})",
                                                       targetPath,
                                                       mx::errorMessage( errc ),
                                                       mx::errorName( errc ) ),
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( m_actMask.rows() != m_flatCommand.rows() || m_actMask.cols() != m_flatCommand.cols() )
    {
        derivedT::template log<text_log>( std::format( "actuaor mask {}x{} is not same size as flag {}x{}",
                                                       m_actMask.rows(),
                                                       m_actMask.cols(),
                                                       m_flatCommand.rows(),
                                                       m_flatCommand.cols() ),
                                          logPrio::LOG_ERROR );

        return -1;
    }

    m_flatCommand *= m_actMask();

    derivedT::template log<text_log>( "loaded flat file " + targetPath );
    m_flatLoaded = true;

    m_flatCurrent = intarget;

    if( m_indiP_flats.find( "default" ) )
    {
        if( m_flatCurrent == "default" )
        {
            m_indiP_flats["default"] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_flats["default"] = pcf::IndiElement::Off;
        }
    }

    for( auto i = m_flatCommands.begin(); i != m_flatCommands.end(); ++i )
    {
        if( !m_indiP_flats.find( i->first ) )
        {
            continue;
        }

        if( i->first == m_flatCurrent )
        {
            m_indiP_flats[i->first] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_flats[i->first] = pcf::IndiElement::Off;
        }
    }

    if( derived().m_indiDriver )
    {
        derived().m_indiDriver->sendSetProperty( m_indiP_flats );
    }

    if( m_flatSet )
    {
        setFlat();
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setFlat( bool update )
{
    if( m_shmimFlat == "" )
    {
        return 0;
    }

    if( !( derived().state() == stateCodes::READY || derived().state() == stateCodes::OPERATING ) )
    {
        derivedT::template log<text_log>( "can not set flat unless DM is READY or OPERATING", logPrio::LOG_WARNING );
        return -1;
    }

    if( ImageStreamIO_openIm( &m_flatImageStream, m_shmimFlat.c_str() ) != 0 )
    {
        derivedT::template log<text_log>( "could not connect to flat channel " + m_shmimFlat, logPrio::LOG_WARNING );
        return -1;
    }

    if( m_flatImageStream.md[0].size[0] != m_dmWidth )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "width mismatch between " + m_shmimFlat + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( m_flatImageStream.md[0].size[1] != m_dmHeight )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "height mismatch between " + m_shmimFlat + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( !m_flatLoaded )
    {
        bool flatSet = m_flatSet;
        m_flatSet    = false; // make sure we don't loop

        if( loadFlat( m_flatCurrent ) < 0 )
        {
            derivedT::template log<text_log>( "error loading flat " + m_flatCurrent, logPrio::LOG_ERROR );
        }

        m_flatSet = flatSet;
    }

    if( !m_flatLoaded )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "no flat loaded", logPrio::LOG_ERROR );
        return -1;
    }

    if( m_flatCommand.rows() != m_dmWidth )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "width mismatch between flat file and configured DM", logPrio::LOG_ERROR );
        return -1;
    }

    if( m_flatCommand.cols() != m_dmHeight )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "height mismatch between flat file and configured DM", logPrio::LOG_ERROR );
        return -1;
    }

    m_flatImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here
    /// anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Copy
    memcpy( m_flatImageStream.array.raw, m_flatCommand.data(), m_dmWidth * m_dmHeight * sizeof( realT ) );

    // Set the time of last write
    clock_gettime( CLOCK_REALTIME, &m_flatImageStream.md->writetime );

    // Set the image acquisition timestamp
    m_flatImageStream.md->atime = m_flatImageStream.md->writetime;

    m_flatImageStream.md->cnt0++;
    m_flatImageStream.md->write = 0;

    // Post the semaphores
    ImageStreamIO_sempost( &m_flatImageStream, -1 );

    m_flatSet = true;

    ImageStreamIO_closeIm( &m_flatImageStream );

    derived().state( stateCodes::OPERATING );

    if( !update )
    {
        derived().updateSwitchIfChanged( m_indiP_setFlat, "toggle", pcf::IndiElement::On, pcf::IndiProperty::Busy );

        derivedT::template log<text_log>( "flat set" );
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::zeroFlat()
{
    if( m_shmimFlat == "" )
    {
        return 0;
    }

    if( !( derived().state() == stateCodes::READY || derived().state() == stateCodes::OPERATING ) )
    {
        derivedT::template log<text_log>( "can not zero flat unless DM is READY or OPERATING", logPrio::LOG_WARNING );
        return -1;
    }

    if( ImageStreamIO_openIm( &m_flatImageStream, m_shmimFlat.c_str() ) != 0 )
    {
        derivedT::template log<text_log>( "could not connect to flat channel " + m_shmimFlat, logPrio::LOG_WARNING );
        return -1;
    }

    if( m_flatImageStream.md[0].size[0] != m_dmWidth )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "width mismatch between " + m_shmimFlat + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( m_flatImageStream.md[0].size[1] != m_dmHeight )
    {
        ImageStreamIO_closeIm( &m_flatImageStream );
        derivedT::template log<text_log>( "height mismatch between " + m_shmimFlat + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    m_flatImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here
    /// anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Zero
    memset( m_flatImageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof( realT ) );

    // Set the time of last write
    clock_gettime( CLOCK_REALTIME, &m_flatImageStream.md->writetime );

    // Set the image acquisition timestamp
    m_flatImageStream.md->atime = m_flatImageStream.md->writetime;

    m_flatImageStream.md->cnt0++;
    m_flatImageStream.md->write = 0;
    ImageStreamIO_sempost( &m_flatImageStream, -1 );

    m_flatSet = false;

    // Post the semaphore
    ImageStreamIO_closeIm( &m_flatImageStream );

    derived().updateSwitchIfChanged( m_indiP_setFlat, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle );

    derivedT::template log<text_log>( "flat zeroed" );

    if( derived().zeroDM() < 0 )
    {
        derivedT::template log<software_error>( { "error from zeroDM" } );
    }

    if( clearSat() < 0 )
    {
        derivedT::template log<software_error>( { "error from clearSat" } );
    }

    derived().state( stateCodes::READY );

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::checkTests()
{
    std::vector<std::string> tfs;

    static bool gfn_logged = false; // tracks if not finding the test path is logged

    mx::error_t errc = mx::ioutils::getFileNames( tfs, m_testPath, "", "", ".fits" );

    if( errc != mx::error_t::noerror )
    {
        if( !gfn_logged )
        {
            derivedT::template log<software_error>(
                { std::format( "error getting test files: {}", mx::errorMessage( errc ) ) } );
        }

        gfn_logged = true;
        return -1;
    }

    gfn_logged = false;

    mx_error_check_rv( errc, -1 );

    for( auto it = m_testCommands.begin(); it != m_testCommands.end(); ++it )
    {
        it->second = "";
    }

    bool changed = false;

    for( size_t n = 0; n < tfs.size(); ++n )
    {
        auto ir =
            m_testCommands.insert( std::pair<std::string, std::string>( mx::ioutils::pathStem( tfs[n] ), tfs[n] ) );

        if( ir.second == true )
        {
            changed = true;
        }
        else
        {
            ir.first->second = tfs[n];
        }
    }

    for( auto it = m_testCommands.begin(); it != m_testCommands.end(); ++it )
    {
        if( it->second == "" )
        {
            changed = true;
            // Erase the current iterator safely, even if the first one.
            auto itdel = it;
            ++it;
            m_testCommands.erase( itdel );
            --it;
        };
    }

    if( changed )
    {
        if( derived().m_indiDriver )
        {
            derived().m_indiDriver->sendDelProperty( m_indiP_tests );
            derived().m_indiNewCallBacks.erase( m_indiP_tests.createUniqueKey() );
        }

        m_indiP_tests = pcf::IndiProperty( pcf::IndiProperty::Switch );
        m_indiP_tests.setDevice( derived().configName() );
        m_indiP_tests.setName( "test" );
        m_indiP_tests.setPerm( pcf::IndiProperty::ReadWrite );
        m_indiP_tests.setState( pcf::IndiProperty::Idle );
        m_indiP_tests.setRule( pcf::IndiProperty::OneOfMany );

        // Add the toggle element initialized to Off
        for( auto it = m_testCommands.begin(); it != m_testCommands.end(); ++it )
        {
            if( it->first == m_testCurrent || m_testCurrent == "" )
            {
                m_indiP_tests.add( pcf::IndiElement( it->first, pcf::IndiElement::On ) );
                m_testCurrent = it->first; // Handles the case when m_testCurrent=="" b/c it was not set in config
            }
            else
            {
                m_indiP_tests.add( pcf::IndiElement( it->first, pcf::IndiElement::Off ) );
            }
        }

        if( m_testDefault != "" )
        {
            if( m_testCurrent == "default" )
            {
                m_indiP_tests.add( pcf::IndiElement( "default", pcf::IndiElement::On ) );
            }
            else
            {
                m_indiP_tests.add( pcf::IndiElement( "default", pcf::IndiElement::Off ) );
            }
        }

        if( derived().registerIndiPropertyNew( m_indiP_tests, st_newCallBack_tests ) < 0 )
        {
#ifndef DM_TEST_NOLOG
            derivedT::template log<software_error>( { "" } );
#endif
            return -1;
        }

        if( derived().m_indiDriver )
        {
            derived().m_indiDriver->sendDefProperty( m_indiP_tests );
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::loadTest( const std::string &intarget )
{
    std::string target = intarget; // store this for later to resolve default next:

    if( target == "default" )
    {
        target = m_testDefault;
    }

    std::string targetPath;

    try
    {
        targetPath = m_testCommands.at( target );
    }
    catch( ... )
    {
        derivedT::template log<text_log>( "test file " + target + " not found", logPrio::LOG_ERROR );
        return -1;
    }

    m_testLoaded = false;
    // load into memory.
    mx::fits::fitsFile<realT> ff;
    mx::error_t               errc = ff.read( m_testCommand, targetPath );

    if( errc != mx::error_t::noerror )
    {
        derivedT::template log<text_log>( std::format( "error reading test file {}: "
                                                       "{} ({})",
                                                       targetPath,
                                                       mx::errorMessage( errc ),
                                                       mx::errorName( errc ) ),
                                          logPrio::LOG_ERROR );
        return -1;
    }

    derivedT::template log<text_log>( "loaded test file " + targetPath );
    m_testLoaded = true;

    m_testCurrent = intarget;

    if( m_indiP_tests.find( "default" ) )
    {
        if( m_testCurrent == "default" )
        {
            m_indiP_tests["default"] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_tests["default"] = pcf::IndiElement::Off;
        }
    }

    for( auto i = m_testCommands.begin(); i != m_testCommands.end(); ++i )
    {
        if( !m_indiP_tests.find( i->first ) )
        {
            continue;
        }

        if( i->first == m_testCurrent )
        {
            m_indiP_tests[i->first] = pcf::IndiElement::On;
        }
        else
        {
            m_indiP_tests[i->first] = pcf::IndiElement::Off;
        }
    }

    if( derived().m_indiDriver )
    {
        derived().m_indiDriver->sendSetProperty( m_indiP_tests );
    }

    if( m_testSet )
    {
        setTest();
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::setTest()
{

    if( m_shmimTest == "" )
    {
        return 0;
    }

    if( ImageStreamIO_openIm( &m_testImageStream, m_shmimTest.c_str() ) != 0 )
    {
        derivedT::template log<text_log>( "could not connect to test channel " + m_shmimTest, logPrio::LOG_WARNING );
        return -1;
    }

    if( m_testImageStream.md->size[0] != m_dmWidth )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "width mismatch between " + m_shmimTest + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( m_testImageStream.md->size[1] != m_dmHeight )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "height mismatch between " + m_shmimTest + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( !m_testLoaded )
    {
        bool testSet = m_testSet;
        m_testSet    = false; // make sure we don't loop

        if( loadTest( m_testCurrent ) < 0 )
        {
            derivedT::template log<text_log>( "error loading test " + m_testCurrent, logPrio::LOG_ERROR );
        }

        m_testSet = testSet;
    }

    if( !m_testLoaded )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "no test loaded", logPrio::LOG_ERROR );
        return -1;
    }

    if( m_testCommand.rows() != m_dmWidth )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "width mismatch between test file and configured DM", logPrio::LOG_ERROR );
        return -1;
    }

    if( m_testCommand.cols() != m_dmHeight )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "height mismatch between test file and configured DM", logPrio::LOG_ERROR );
        return -1;
    }

    m_testImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here
    /// anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Copy
    memcpy( m_testImageStream.array.raw, m_testCommand.data(), m_dmWidth * m_dmHeight * sizeof( realT ) );

    // Set the time of last write
    clock_gettime( CLOCK_REALTIME, &m_testImageStream.md->writetime );

    // Set the image acquisition timestamp
    m_testImageStream.md->atime = m_testImageStream.md->writetime;

    m_testImageStream.md->cnt0++;
    m_testImageStream.md->write = 0;
    ImageStreamIO_sempost( &m_testImageStream, -1 );

    m_testSet = true;

    // Post the semaphore
    ImageStreamIO_closeIm( &m_testImageStream );

    derived().updateSwitchIfChanged( m_indiP_setTest, "toggle", pcf::IndiElement::On, pcf::IndiProperty::Busy );

    derivedT::template log<text_log>( "test set" );

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::zeroTest()
{
    if( m_shmimTest == "" )
    {
        return 0;
    }

    if( ImageStreamIO_openIm( &m_testImageStream, m_shmimTest.c_str() ) != 0 )
    {
        derivedT::template log<text_log>( "could not connect to test channel " + m_shmimTest, logPrio::LOG_WARNING );
        return -1;
    }

    if( m_testImageStream.md[0].size[0] != m_dmWidth )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "width mismatch between " + m_shmimTest + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    if( m_testImageStream.md[0].size[1] != m_dmHeight )
    {
        ImageStreamIO_closeIm( &m_testImageStream );
        derivedT::template log<text_log>( "height mismatch between " + m_shmimTest + " and configured DM",
                                          logPrio::LOG_ERROR );
        return -1;
    }

    m_testImageStream.md->write = 1;

    ///\todo we are assuming that dmXXcomYY is not a cube.  This might be true, but we should add cnt1 handling here
    /// anyway.  With bounds checks b/c not everyone handles cnt1 properly.
    // Zero
    memset( m_testImageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof( realT ) );

    // Set the time of last write
    clock_gettime( CLOCK_REALTIME, &m_testImageStream.md->writetime );

    // Set the image acquisition timestamp
    m_testImageStream.md->atime = m_testImageStream.md->writetime;

    m_testImageStream.md->cnt0++;
    m_testImageStream.md->write = 0;

    // Post the semaphore
    ImageStreamIO_sempost( &m_testImageStream, -1 );

    m_testSet = false;

    ImageStreamIO_closeIm( &m_testImageStream );

    derived().updateSwitchIfChanged( m_indiP_setTest, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle );

    derivedT::template log<text_log>( "test zeroed" );

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::zeroAll( bool nosem )
{
    if( derived().m_shmimName == "" )
    {
        return 0;
    }

    int auxiliaryError = 0;

    for( int n = 0; n < m_numChannels; ++n )
    {
        char nstr[16];
        snprintf( nstr, sizeof( nstr ), "%02d", n );
        // Clear the persistent source before its destination; sources are not extra main channels.
        for( const char *suffix : { "_aux", "" } )
        {
            bool        auxiliary = suffix[0] != '\0';
            std::string shmimN    = derived().m_shmimName + nstr + suffix;

            if( auxiliary )
            {
                // Avoid ImageStreamIO's error output for normally absent auxiliary sources.
                char filename[200];

                if( ImageStreamIO_filename( filename, sizeof( filename ), shmimN.c_str() ) != IMAGESTREAMIO_SUCCESS )
                {
                    derivedT::template log<software_error>(
                        { "could not construct auxiliary channel path " + shmimN } );
                    auxiliaryError = -1;
                    continue;
                }

                int fd = open( filename, O_RDWR );

                if( fd < 0 )
                {
                    if( errno == ENOENT )
                    {
                        continue;
                    }

                    derivedT::template log<software_error>( { errno, "could not open auxiliary channel " + shmimN } );
                    auxiliaryError = -1;
                    continue;
                }

                close( fd );
            }

            IMAGE imageStream{};

            if( ImageStreamIO_openIm( &imageStream, shmimN.c_str() ) != 0 )
            {
                derivedT::template log<text_log>( "could not connect to channel " + shmimN, logPrio::LOG_WARNING );

                if( auxiliary )
                {
                    auxiliaryError = -1;
                    continue;
                }

                continue;
            }

            if( imageStream.md->size[0] != m_dmWidth || imageStream.md->size[1] != m_dmHeight ||
                ( auxiliary && ( imageStream.md->location != -1 || imageStream.md->naxis != 2 ||
                                 imageStream.md->datatype != m_dmDataType ) ) )
            {
                ImageStreamIO_closeIm( &imageStream );
                derivedT::template log<text_log>( "memory location, shape or data type mismatch between " + shmimN +
                                                      " and configured DM",
                                                  logPrio::LOG_ERROR );

                if( auxiliary )
                {
                    auxiliaryError = -1;
                    continue;
                }

                derived().updateSwitchIfChanged( m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE );
                return -1;
            }

            imageStream.md->write = 1;
            memset( imageStream.array.raw, 0, m_dmWidth * m_dmHeight * sizeof( realT ) );

            clock_gettime( CLOCK_REALTIME, &imageStream.md->writetime );
            imageStream.md->atime = imageStream.md->writetime;
            imageStream.md->cnt0++;
            imageStream.md->write = 0;

            if( !auxiliary && n == m_numChannels - 1 && !nosem )
            {
                ImageStreamIO_sempost( &imageStream, -1 );
            }

            ImageStreamIO_closeIm( &imageStream );
        }
    }

    if( auxiliaryError == 0 )
    {
        derivedT::template log<text_log>( "all channels zeroed", logPrio::LOG_NOTICE );
    }
    else
    {
        derivedT::template log<text_log>( "main channels zeroed; not all auxiliary sources could be cleared",
                                          logPrio::LOG_ERROR );
    }

    derived().updateSwitchIfChanged( m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE );

    // Also cleanup flat and test
    m_flatSet = false;
    derived().updateSwitchIfChanged( m_indiP_setFlat, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle );

    if( derived().state() == stateCodes::OPERATING )
    {
        derived().state( stateCodes::READY );
    }

    // Also cleanup flat and test
    m_testSet = false;
    derived().updateSwitchIfChanged( m_indiP_setTest, "toggle", pcf::IndiElement::Off, pcf::IndiProperty::Idle );

    int rv;

    if( ( rv = clearSat() ) < 0 )
    {
        derivedT::template log<software_error>( { errno, rv, "Error from clearSat" } );
        return rv;
    }

    return auxiliaryError;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::makeDelta()
{
    if( m_notDeltas.size() == 0 )
    {
        return 0;
    }

    m_totalFlat = ( *m_channels[m_notDeltas[0]] )();

    for( size_t n = 1; n < m_notDeltas.size(); ++n )
    {
        m_totalFlat += ( *m_channels[m_notDeltas[n]] )();
    }

    m_outputDelta = m_outputShape() - m_totalFlat; // this posts and everything

    m_totalDelta = ( *m_channels[m_deltas[0]] )();

    for( size_t n = 1; n < m_deltas.size(); ++n )
    {
        m_totalDelta += ( *m_channels[m_deltas[n]] )();
    }

    m_outputDiff = m_totalDelta - m_outputDelta();

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::stageDelta()
{
    return stageDelta( m_outputShape(), m_stagedDelta, m_stagedDiff, m_deltaStaged );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::stageDelta( Eigen::Ref<const mx::improc::eigenImage<realT>> shape,
                                     mx::improc::eigenImage<realT>                  &delta,
                                     mx::improc::eigenImage<realT>                  &diff,
                                     bool                                           &staged )
{
    staged = false;

    if( m_notDeltas.empty() )
    {
        return 0;
    }

    if( m_deltas.empty() || delta.rows() != m_dmWidth || delta.cols() != m_dmHeight || diff.rows() != m_dmWidth ||
        diff.cols() != m_dmHeight || m_totalFlat.rows() != m_dmWidth || m_totalFlat.cols() != m_dmHeight ||
        m_totalDelta.rows() != m_dmWidth || m_totalDelta.cols() != m_dmHeight || shape.rows() != m_dmWidth ||
        shape.cols() != m_dmHeight )
    {
        return -1;
    }

    for( const auto &indices : { &m_notDeltas, &m_deltas } )
    {
        for( const size_t index : *indices )
        {
            if( index >= m_channels.size() || m_channels[index] == nullptr ||
                ( *m_channels[index] )().rows() != m_dmWidth || ( *m_channels[index] )().cols() != m_dmHeight )
            {
                return -1;
            }
        }
    }

    m_totalFlat = ( *m_channels[m_notDeltas[0]] )();

    for( size_t n = 1; n < m_notDeltas.size(); ++n )
    {
        m_totalFlat += ( *m_channels[m_notDeltas[n]] )();
    }

    m_totalDelta = ( *m_channels[m_deltas[0]] )();

    for( size_t n = 1; n < m_deltas.size(); ++n )
    {
        m_totalDelta += ( *m_channels[m_deltas[n]] )();
    }

    delta  = shape - m_totalFlat;
    diff   = m_totalDelta - delta;
    staged = true;

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::publishDelta()
{
    if( m_deltaStaged )
    {
        m_outputDelta = m_stagedDelta;
        m_outputDiff  = m_stagedDiff;
        m_deltaStaged = false;
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::clearSat()
{
    std::unique_lock<std::mutex> pipelineLock( m_pipelineSatMutex, std::defer_lock );

    if( m_pipelineSatEnabled.load( std::memory_order_acquire ) )
    {
        pipelineLock.lock();
        m_pipelineSaturation.clear();
        m_pipelineSatWindow.setZero();
        ++m_pipelineSatGeneration;
    }

    if( m_shmimSat == "" || m_dmWidth == 0 || m_dmHeight == 0 )
    {
        return 0;
    }

    IMAGE imageStream;

    std::vector<std::string> sats = { m_shmimSat, m_shmimSatPerc };

    for( size_t n = 0; n < sats.size(); ++n )
    {
        std::string shmimN = sats[n];

        if( ImageStreamIO_openIm( &imageStream, shmimN.c_str() ) != 0 )
        {
            derivedT::template log<text_log>( "could not connect to sat map " + shmimN, logPrio::LOG_WARNING );
            return 0;
        }

        if( imageStream.md->size[0] != m_dmWidth )
        {
            ImageStreamIO_closeIm( &imageStream );
            derivedT::template log<text_log>( "width mismatch between " + shmimN + " and configured DM",
                                              logPrio::LOG_ERROR );
            derived().updateSwitchIfChanged( m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE );
            return -1;
        }

        if( imageStream.md->size[1] != m_dmHeight )
        {
            ImageStreamIO_closeIm( &imageStream );
            derivedT::template log<text_log>( "height mismatch between " + shmimN + " and configured DM",
                                              logPrio::LOG_ERROR );
            derived().updateSwitchIfChanged( m_indiP_zeroAll, "request", pcf::IndiElement::Off, INDI_IDLE );
            return -1;
        }

        imageStream.md->write = 1;
        memset( imageStream.array.raw, 0, m_dmWidth * m_dmHeight * ImageStreamIO_typesize( imageStream.md->datatype ) );

        clock_gettime( CLOCK_REALTIME, &imageStream.md->writetime );

        // Set the image acquisition timestamp
        imageStream.md->atime = imageStream.md->writetime;

        imageStream.md->cnt0++;
        imageStream.md->write = 0;
        ImageStreamIO_sempost( &imageStream, -1 );

        ImageStreamIO_closeIm( &imageStream );
    }

    m_accumSatMap.setZero();
    m_instSatMap.setZero();

    return 0;
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::satThreadStart( dm *d )
{
    d->satThreadExec();
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::satThreadExec()
{
    // Get the thread PID immediately so the caller can return.
    m_satThreadID = syscall( SYS_gettid );

    // Wait for the thread starter to finish initializing this thread.
    while( m_satThreadInit == true && derived().shutdown() == 0 )
    {
        sleep( 1 );
    }

    if( derived().shutdown() )
    {
        return;
    }

    uint32_t imsize[3] = { 0, 0, 0 };

    // Check for allocation to have happened.
    while( ( m_shmimSat == "" || m_accumSatMap.rows() == 0 || m_accumSatMap.cols() == 0 ) && !derived().shutdown() )
    {
        sleep( 1 );
    }

    if( derived().shutdown() )
    {
        return;
    }

    imsize[0] = m_dmWidth;
    imsize[1] = m_dmHeight;
    imsize[2] = 1;

    ImageStreamIO_createIm_gpu( &m_satImageStream,
                                m_shmimSat.c_str(),
                                3,
                                imsize,
                                IMAGESTRUCT_UINT8,
                                -1,
                                1,
                                IMAGE_NB_SEMAPHORE,
                                0,
                                CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                0 );
    ImageStreamIO_createIm_gpu( &m_satPercImageStream,
                                m_shmimSatPerc.c_str(),
                                3,
                                imsize,
                                IMAGESTRUCT_FLOAT,
                                -1,
                                1,
                                IMAGE_NB_SEMAPHORE,
                                0,
                                CIRCULAR_BUFFER | ZAXIS_TEMPORAL,
                                0 );

    bool opened = true;

    m_satImageStream.md->cnt1     = 0;
    m_satPercImageStream.md->cnt1 = 0;

    // This is the working memory for making the 1/0 mask out of m_accumSatMap
    mx::improc::eigenImage<uint8_t> satmap( m_dmWidth, m_dmHeight );

    uint64_t naccum             = 0;
    uint64_t pipelineGeneration = 0;
    double   t_accumst          = mx::sys::get_curr_time();

    // This is the main image grabbing loop.
    while( !derived().shutdown() )
    {
        // Get timespec for sem_timedwait
        timespec ts;

        if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
        {
            derivedT::template log<software_critical>( { errno, 0, "clock_gettime" } );
            return;
        }

        ts.tv_sec += 1;

        // Wait on semaphore
        if( sem_timedwait( &m_satSemaphore, &ts ) == 0 )
        {
            const bool                   pipeline = m_pipelineSatEnabled.load( std::memory_order_acquire );
            std::unique_lock<std::mutex> pipelineLock( m_pipelineSatMutex, std::defer_lock );

            if( pipeline )
            {
                pipelineLock.lock();

                if( pipelineGeneration != m_pipelineSatGeneration )
                {
                    pipelineGeneration = m_pipelineSatGeneration;
                    naccum             = 0;
                    t_accumst          = mx::sys::get_curr_time();
                }

                const uint64_t samples =
                    m_pipelineSaturation.drain( m_pipelineSatWindow.data(), size_t( m_pipelineSatWindow.size() ) );

                if( samples == 0 )
                {
                    continue;
                }

                naccum += samples;
            }
            else
            {
                for( int rr = 0; rr < m_instSatMap.rows(); ++rr )
                {
                    for( int cc = 0; cc < m_instSatMap.cols(); ++cc )
                    {
                        m_accumSatMap( rr, cc ) += m_instSatMap( rr, cc );
                    }
                }

                ++naccum;
            }

            // If less than avg int --> go back and wait again
            if( mx::sys::get_curr_time( ts ) - t_accumst < m_satAvgInt / 1000.0 )
            {
                continue;
            }

            // If greater than avg int --> calc stats, write to streams.
            m_overSatAct = 0;

            for( int rr = 0; rr < m_instSatMap.rows(); ++rr )
            {
                for( int cc = 0; cc < m_instSatMap.cols(); ++cc )
                {
                    const uint64_t accumulated = pipeline ? m_pipelineSatWindow( rr, cc ) : m_accumSatMap( rr, cc );
                    m_satPercMap( rr, cc ) =
                        pipeline ? float( accumulated ) / float( naccum ) : m_accumSatMap( rr, cc ) / naccum;

                    if( m_satPercMap( rr, cc ) >= m_percThreshold )
                    {
                        ++m_overSatAct;
                    }

                    satmap( rr, cc ) = accumulated > 0;
                }
            }

            // Check of the number of actuators saturated above the percent threshold is greater than the number
            // threshold if it is, increment the counter
            if( m_overSatAct / ( m_satPercMap.rows() * m_satPercMap.cols() * 0.75 ) > m_intervalSatThreshold )
            {
                ++m_intervalSatExceeds;
            }
            else
            {
                m_intervalSatExceeds = 0;
            }

            // If enough consecutive intervals exceed the count threshold, we trigger
            if( m_intervalSatExceeds >= m_intervalSatCountThreshold )
            {
                m_intervalSatTrip = true;
            }

            m_satImageStream.md->write     = 1;
            m_satPercImageStream.md->write = 1;

            memcpy( m_satImageStream.array.raw, satmap.data(), m_dmWidth * m_dmHeight * sizeof( uint8_t ) );
            memcpy( m_satPercImageStream.array.raw, m_satPercMap.data(), m_dmWidth * m_dmHeight * sizeof( float ) );

            // Set the time of last write
            clock_gettime( CLOCK_REALTIME, &m_satImageStream.md->writetime );
            m_satPercImageStream.md->writetime = m_satImageStream.md->writetime;

            // Set the image acquisition timestamp
            m_satImageStream.md->atime     = m_satImageStream.md->writetime;
            m_satPercImageStream.md->atime = m_satPercImageStream.md->writetime;

            // Update cnt1
            m_satImageStream.md->cnt1     = 0;
            m_satPercImageStream.md->cnt1 = 0;

            // Update cnt0
            m_satImageStream.md->cnt0++;
            m_satPercImageStream.md->cnt0++;

            m_satImageStream.writetimearray[0] = m_satImageStream.md->writetime;
            m_satImageStream.atimearray[0]     = m_satImageStream.md->atime;
            m_satImageStream.cntarray[0]       = m_satImageStream.md->cnt0;

            m_satPercImageStream.writetimearray[0] = m_satPercImageStream.md->writetime;
            m_satPercImageStream.atimearray[0]     = m_satPercImageStream.md->atime;
            m_satPercImageStream.cntarray[0]       = m_satPercImageStream.md->cnt0;

            // And post
            m_satImageStream.md->write = 0;
            ImageStreamIO_sempost( &m_satImageStream, -1 );

            m_satPercImageStream.md->write = 0;
            ImageStreamIO_sempost( &m_satPercImageStream, -1 );

            m_accumSatMap.setZero();

            if( pipeline )
            {
                m_pipelineSatWindow.setZero();
            }

            naccum    = 0;
            t_accumst = mx::sys::get_curr_time( ts );
        }
        else
        {
            // Check for why we timed out
            if( errno == EINTR )
            {
                break; // This indicates signal interrupted us, time to restart or shutdown, loop will exit normally if
                       // flags set.
            }

            // ETIMEDOUT just means we should wait more.
            // Otherwise, report an error.
            if( errno != ETIMEDOUT )
            {
                derivedT::template log<software_error>( { errno, "sem_timedwait" } );
                break;
            }
        }
    }

    if( opened )
    {
        ImageStreamIO_destroyIm( &m_satImageStream );

        ImageStreamIO_destroyIm( &m_satPercImageStream );
    }
}

template <class derivedT, typename realT>
void dm<derivedT, realT>::intervalSatTrip()
{
    if( m_satTriggerDevice.size() > 0 && m_satTriggerProperty.size() == m_satTriggerDevice.size() )
    {
        for( size_t n = 0; n < m_satTriggerDevice.size(); ++n )
        {
            // We just silently fail
            try
            {
                pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

                ipFreq.setDevice( m_satTriggerDevice[n] );
                ipFreq.setName( m_satTriggerProperty[n] );
                ipFreq.add( pcf::IndiElement( "toggle" ) );
                ipFreq["toggle"] = pcf::IndiElement::Off;
                derived().sendNewProperty( ipFreq );

                derivedT::template log<text_log>( "DM saturation threshold exceeded.  Loop opened.",
                                                  logPrio::LOG_WARNING );
            }
            catch( ... )
            {
            }
        }
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::updateINDI()
{
    if( !derived().m_indiDriver )
    {
        return 0;
    }

    updateDMCompletionINDI();

    const auto &summary = m_dmTiming.summary();
    auto        state   = summary.fgValid ? pcf::IndiProperty::Ok : pcf::IndiProperty::Idle;
    const auto &a       = summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::receiptInterval )];
    const auto &w       = summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::sendInterval )];
    const auto &wa      = summary.metrics[MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::receiptSend )];
    indi::updateIfChanged<double>( m_indiP_dmTiming,
                                   { "receipt_fps",
                                     "receipt_min",
                                     "receipt_max",
                                     "receipt_jitter",
                                     "receipt_count",
                                     "send_fps",
                                     "send_min",
                                     "send_max",
                                     "send_jitter",
                                     "send_count",
                                     "receipt_send",
                                     "receipt_send_jitter",
                                     "receipt_send_min",
                                     "receipt_send_max",
                                     "receipt_send_count" },
                                   { a.valid && a.mean > 0 ? 1.0 / a.mean : -1,
                                     a.count ? a.min : -1,
                                     a.count ? a.max : -1,
                                     a.valid ? a.stddev : -1,
                                     static_cast<double>( a.count ),
                                     w.valid && w.mean > 0 ? 1.0 / w.mean : -1,
                                     w.count ? w.min : -1,
                                     w.count ? w.max : -1,
                                     w.valid ? w.stddev : -1,
                                     static_cast<double>( w.count ),
                                     wa.count ? wa.mean : -1,
                                     wa.valid ? wa.stddev : -1,
                                     wa.count ? wa.min : -1,
                                     wa.count ? wa.max : -1,
                                     static_cast<double>( wa.count ) },
                                   derived().m_indiDriver,
                                   state );

    std::vector<std::string> names;
    std::vector<double>      values;

    for( size_t i = MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::dispatch );
         i < MagAOX::dmTiming::metricIndex( MagAOX::dmTiming::Metric::count );
         ++i )
    {
        std::string name   = MagAOX::dmTiming::metricName( static_cast<MagAOX::dmTiming::Metric>( i ) );
        const auto &metric = summary.metrics[i];

        for( const auto *field : { "mean", "jitter", "min", "max", "count", "span" } )
        {
            names.push_back( name + '_' + field );
        }

        values.insert( values.end(),
                       { metric.count ? metric.mean : -1,
                         metric.valid ? metric.stddev : -1,
                         metric.count ? metric.min : -1,
                         metric.count ? metric.max : -1,
                         static_cast<double>( metric.count ),
                         metric.count ? metric.span : -1 } );
    }

    indi::updateIfChanged<double>( m_indiP_dmStages, names, values, derived().m_indiDriver, state );

    const auto &q = summary.total;
    indi::updateIfChanged<double>( m_indiP_dmTimingQuality,
                                   { "window",      "span",    "age",    "capacity",  "queued",   "session",
                                     "sequence",    "enabled", "valid",  "events",    "receipts", "submitted",
                                     "failed",      "partial", "noSend", "synthetic", "rejected", "duplicates",
                                     "counterGaps", "invalid", "dropped" },
                                   { summary.windowSec,
                                     summary.span,
                                     summary.age,
                                     static_cast<double>( summary.capacity ),
                                     static_cast<double>( summary.queued ),
                                     static_cast<double>( summary.session ),
                                     static_cast<double>( summary.sequence ),
                                     static_cast<double>( summary.enabled ),
                                     static_cast<double>( summary.fgValid ),
                                     static_cast<double>( q.events ),
                                     static_cast<double>( q.receipts ),
                                     static_cast<double>( q.submitted ),
                                     static_cast<double>( q.failed ),
                                     static_cast<double>( q.partial ),
                                     static_cast<double>( q.noSend ),
                                     static_cast<double>( q.synthetic ),
                                     static_cast<double>( q.rejected ),
                                     static_cast<double>( q.duplicates ),
                                     static_cast<double>( q.counterGaps ),
                                     static_cast<double>( q.invalid ),
                                     static_cast<double>( q.dropped ) },
                                   derived().m_indiDriver,
                                   state );

    const auto &delivery = summary.delivery;
    const auto &d        = delivery.total;
    indi::updateIfChanged<double>( m_indiP_dmTimingDelivery,
                                   { "enabled",
                                     "clock",
                                     "bracketUs",
                                     "attempted",
                                     "qualified",
                                     "unavailable",
                                     "invalid",
                                     "wideBracket",
                                     "clockDiscontinuity",
                                     "futurePublication",
                                     "identityRejected" },
                                   { delivery.enabled ? 1.0 : 0.0,
                                     static_cast<double>( delivery.clock ),
                                     delivery.bracketLimitNs * 0.001,
                                     static_cast<double>( d.attempted ),
                                     static_cast<double>( d.qualified ),
                                     static_cast<double>( d.unavailable ),
                                     static_cast<double>( d.invalid ),
                                     static_cast<double>( d.wideBracket ),
                                     static_cast<double>( d.clockDiscontinuity ),
                                     static_cast<double>( d.futurePublication ),
                                     static_cast<double>( d.identityRejected ) },
                                   derived().m_indiDriver,
                                   delivery.enabled && d.qualified ? pcf::IndiProperty::Ok : pcf::IndiProperty::Idle );
    const auto deliveryClock =
        delivery.clock == MagAOX::dmTiming::PublicationClock::tai
            ? "CLOCK_TAI"
            : ( delivery.clock == MagAOX::dmTiming::PublicationClock::realtime ? "CLOCK_REALTIME" : "disabled" );
    indi::updateIfChanged(
        m_indiP_dmTimingState, "deliveryClock", std::string( deliveryClock ), derived().m_indiDriver );

    const char *states[] = { "disabled", "warmingUp", "valid", "stale", "unavailable" };
    indi::updateIfChanged( m_indiP_dmTimingState,
                           "state",
                           std::string( states[static_cast<size_t>( summary.state )] ),
                           derived().m_indiDriver,
                           state );
    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_init( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_init( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_init( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_init.createUniqueKey() )
    {
        return derivedT::template log<software_error, -1>( { "wrong INDI-P in callback" } );
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        int rv = baseInitDM();

        if( rv < 0 )
        {
            return derivedT::template log<software_error, -1>( { "error from initDM in INDI callback" } );
        }
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_zero( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_zero( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_zero( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_zero.createUniqueKey() )
    {
        return derivedT::template log<software_error, -1>( { "wrong INDI-P in callback" } );
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        return derived().zeroDM();
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_release( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_release( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_release( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_release.createUniqueKey() )
    {
        return derivedT::template log<software_error, -1>( { "wrong INDI-P in callback" } );
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        return baseReleaseDM();
    }

    return 0;
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_flats( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_flats( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_flats( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_flats.createUniqueKey() )
    {
        derivedT::template log<software_error>( { "invalid indi property received" } );
        return -1;
    }

    std::string newFlat;

    if( ipRecv.find( "default" ) )
    {
        if( ipRecv["default"].getSwitchState() == pcf::IndiElement::On )
        {
            newFlat = "default";
        }
    }

    // always do this to check for error:
    for( auto i = m_flatCommands.begin(); i != m_flatCommands.end(); ++i )
    {
        if( !ipRecv.find( i->first ) )
        {
            continue;
        }

        if( ipRecv[i->first].getSwitchState() == pcf::IndiElement::On )
        {
            if( newFlat != "" )
            {
                derivedT::template log<text_log>( "More than one flat selected", logPrio::LOG_ERROR );
                return -1;
            }

            newFlat = i->first;
        }
    }

    if( newFlat == "" )
    {
        return 0;
    }

    return loadFlat( newFlat );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_setFlat( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_setFlat( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_setFlat( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_setFlat.createUniqueKey() )
    {
        return derivedT::template log<software_error, -1>( { "wrong INDI-P in callback" } );
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"] == pcf::IndiElement::On )
    {
        return setFlat();
    }
    else
    {
        return zeroFlat();
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_tests( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_tests( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_tests( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_tests.createUniqueKey() )
    {
        derivedT::template log<software_error>( { "invalid indi property received" } );
        return -1;
    }

    std::string newTest;

    if( ipRecv.find( "default" ) )
    {
        if( ipRecv["default"].getSwitchState() == pcf::IndiElement::On )
        {
            newTest = "default";
        }
    }

    // always do this to check for error:
    for( auto i = m_testCommands.begin(); i != m_testCommands.end(); ++i )
    {
        if( !ipRecv.find( i->first ) )
        {
            continue;
        }

        if( ipRecv[i->first].getSwitchState() == pcf::IndiElement::On )
        {
            if( newTest != "" )
            {
                derivedT::template log<text_log>( "More than one test selected", logPrio::LOG_ERROR );
                return -1;
            }

            newTest = i->first;
        }
    }

    if( newTest == "" )
    {
        return 0;
    }

    return loadTest( newTest );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_setTest( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_setTest( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_setTest( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_setTest.createUniqueKey() )
    {
        return derivedT::template log<software_error, -1>( { "wrong INDI-P in callback" } );
    }

    if( !ipRecv.find( "toggle" ) )
    {
        return 0;
    }

    if( ipRecv["toggle"] == pcf::IndiElement::On )
    {
        return setTest();
    }
    else
    {
        return zeroTest();
    }
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::st_newCallBack_zeroAll( void *app, const pcf::IndiProperty &ipRecv )
{
    return static_cast<derivedT *>( app )->newCallBack_zeroAll( ipRecv );
}

template <class derivedT, typename realT>
int dm<derivedT, realT>::newCallBack_zeroAll( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.createUniqueKey() != m_indiP_zeroAll.createUniqueKey() )
    {
        return derivedT::template log<software_error, -1>( { "wrong INDI-P in callback" } );
    }

    if( !ipRecv.find( "request" ) )
    {
        return 0;
    }

    if( ipRecv["request"].getSwitchState() == pcf::IndiElement::On )
    {
        indi::updateSwitchIfChanged(
            m_indiP_zeroAll, "request", pcf::IndiElement::On, derived().m_indiDriver, INDI_BUSY );

        std::lock_guard<std::mutex> guard( derived().m_indiMutex );
        return zeroAll();
    }

    return 0;
}

/// Call dmT::setupConfig with error checking for dm
/**
 * \param cfig the application configurator
 */
#define DM_SETUP_CONFIG( cfig )                                                                                        \
    if( dmT::setupConfig( cfig ) < 0 )                                                                                 \
    {                                                                                                                  \
        log<software_error>( { "Error from dmT::setupConfig" } );                                                      \
        m_shutdown = true;                                                                                             \
        return;                                                                                                        \
    }

/// Call dmT::loadConfig with error checking for dm
/** This must be inside a function that returns int, e.g. the standard loadConfigImpl.
 * \param cfig the application configurator
 */
#define DM_LOAD_CONFIG( cfig )                                                                                         \
    if( dmT::loadConfig( cfig ) < 0 )                                                                                  \
    {                                                                                                                  \
        return log<software_error, -1>( { "Error from dmT::loadConfig" } );                                            \
    }

/// Call shmimMonitorT::appStartup with error checking for dm
#define DM_APP_STARTUP                                                                                                 \
    if( dmT::appStartup() < 0 )                                                                                        \
    {                                                                                                                  \
        return log<software_error, -1>( { "Error from dmT::appStartup" } );                                            \
    }

/// Call dmT::appLogic with error checking for dm
#define DM_APP_LOGIC                                                                                                   \
    if( dmT::appLogic() < 0 )                                                                                          \
    {                                                                                                                  \
        return log<software_error, -1>( { "Error from dmT::appLogic" } );                                              \
    }

/// Call dmT::updateINDI with error checking for dm
#define DM_UPDATE_INDI                                                                                                 \
    if( dmT::updateINDI() < 0 )                                                                                        \
    {                                                                                                                  \
        return log<software_error, -1>( { "Error from dmT::updateINDI" } );                                            \
    }

/// Call dmT::appShutdown with error checking for dm
#define DM_APP_SHUTDOWN                                                                                                \
    if( dmT::appShutdown() < 0 )                                                                                       \
    {                                                                                                                  \
        return log<software_error, -1>( { "Error from dmT::appShutdown" } );                                           \
    }

} // namespace dev
} // namespace app
} // namespace MagAOX
#endif
