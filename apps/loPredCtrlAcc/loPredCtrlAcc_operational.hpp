/** \file loPredCtrlAcc_operational.hpp
 * \brief Closed-loop DDSPC: WFS-rate standard fit and per-DM-slot ACC-assisted fit, with tagged auxiliary DM delivery.
 * \ingroup app_files
 */
#ifndef loPredCtrlAcc_operational_hpp
#define loPredCtrlAcc_operational_hpp

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <semaphore.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <mx/ioutils/fits/fitsFile.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "../../libMagAOX/libMagAOX.hpp"
#ifndef MAGAOX_CURRENT_SHA1
    #include "../../magaox_git_version.h"
#endif
#include "../../libMagAOX/app/syncoOutput.hpp"
#include "adc_reader.hpp"
#include "multirate_ddspc.hpp"
#include "native_pipeline.hpp"

namespace MagAOX::app
{
/// Operational loPredCtrl workflow; upstream owns ACC/WFS synchronization and the DM owns its scheduler.
/** Standard mode is unchanged loPredCtrl DDSPC: one decision per WFS frame on the WFS monitor thread, held on the q DM
 * slots of the frame, trained only while standard is selected. The original's commands were increments integrated by
 * the main loop's mfilt; by default this app applies them as absolute positions. The optional [integrator] restores
 * the mfilt law, s = mult*s - gain*delta, and trains the fit on the deltas
 * (agents/plans/2026-11/ddspc-standard-integrator.md).
 *
 * Assisted mode (agents/plans/2026-11/ddspc-multirate-8khz-assisted.md) recalculates every DM slot on the DM worker.
 * The WFS thread appends each frame to DDSPC::MultirateDDSPC, publishes the frame's origin, then performs the single
 * shared QR-RLS update per genuine WFS target and refreshes the phase coefficients and slot gains, which reach the
 * worker through a triple buffer. Each slot uses the newest r-per-frame ACC samples counted from the frame's anchor
 * and the commands the scheduler actually copied earlier in the frame; nothing waits for learning.
 *
 * Slot ordinals are scheduler cycle*steps+step. A frame's slots follow the newest slot copied when the WFS frame
 * arrived. Training uses the commands copied since the previous decision; missed or discarded slots retain the
 * previous copy, as synco does.
 *
 * Main-loop modal output, as the original loPredCtrl: each new WFS frame is republished through the frameGrabber.
 * With prediction off it is copied unchanged, as if DDSPC were absent. While DDSPC owns tip/tilt (prediction accepted,
 * until its own-channel zero is confirmed after prediction stops) modes 0 and 1 are set to zero and every other mode
 * is copied unchanged, so the main loop keeps correcting higher-order modes while DDSPC alone drives tip/tilt. The WFS
 * input stream and the DDSPC measurement are never modified.
 */
class loPredCtrlAcc_operational : public MagAOXApp<true>,
                                  public dev::shmimMonitor<loPredCtrlAcc_operational>,
                                  public dev::frameGrabber<loPredCtrlAcc_operational>,
                                  public dev::telemeter<loPredCtrlAcc_operational>,
                                  protected dev::dmSyncopationProducer<loPredCtrlAcc_operational>
{
    friend class dev::shmimMonitor<loPredCtrlAcc_operational>;
    friend class dev::frameGrabber<loPredCtrlAcc_operational>;
    friend class dev::telemeter<loPredCtrlAcc_operational>;
    friend class dev::dmSyncopationProducer<loPredCtrlAcc_operational>;
    using shmimMonitorT          = dev::shmimMonitor<loPredCtrlAcc_operational>;
    using frameGrabberT          = dev::frameGrabber<loPredCtrlAcc_operational>;
    using telemeterT             = dev::telemeter<loPredCtrlAcc_operational>;
    using dmSyncopationProducerT = dev::dmSyncopationProducer<loPredCtrlAcc_operational>;

  public:
    /// The modal output is a vector and cannot be flipped.
    static constexpr bool c_frameGrabber_flippable = false;

  protected:
    /// Common model parameters; the RLS/QR implementation is shared with the existing controller.
    DDSPC::ShadowConfig m_settings;

    /// Existing full-image calibration, or the same modal cube used by the operational modal-to-DM path.
    DDSPC::NativeCalibration m_calibration;
    std::string              m_calibrationFile, m_modeCube, m_mask;

    std::uint32_t m_calWidth{ 0 }, m_calHeight{ 0 }; ///< Exact native FITS plane dimensions when using a cube.

    /// Exact owned auxiliary source; must be a scheduler `_aux` image, never a combined DM destination.
    std::string m_channel{ "dm01disp10_aux" };

    /// Independent raw ADC decoder; absence or invalid calibration affects only the assisted model.
    bool                              m_accelEnabled{ true };
    DDSPC::AdcConfig                  m_adcConfig;
    std::unique_ptr<DDSPC::AdcReader> m_adc;
    DDSPC::AdcCursor                  m_adcCursor;
    std::array<DDSPC::AdcSample, 128> m_adcHistory{};
    unsigned                          m_adcCount{ 0 }, m_adcNext{ 0 };
    std::uint64_t m_adcAnchor{ 0 };  ///< Counter of the newest sample in the latest WFS-thread window.
    std::uint64_t m_adcSegment{ 0 }; ///< Source generation and segment of that window.

    /// Existing producer lock diagnostic; there are no local phase estimates or counter anchors.
    std::string       m_syncDevice{ "acc000" }, m_syncProperty{ "timingDiag" };
    std::atomic<bool> m_syncLocked{ false };

    /// Latest upstream lock report, for qualification diagnostics only (-1 or NaN when absent).
    std::atomic<int>          m_upstreamEnabled{ -1 }, m_upstreamLocked{ -1 }; ///< mode_code/enabled, delay_lock/valid.
    std::atomic<double>       m_upstreamPhaseUs{ std::nan( "" ) };             ///< timingDiag delay_phase_error_us.
    std::atomic<double>       m_upstreamLatencyUs{ std::nan( "" ) };           ///< timingDiag read_latency_error_us.
    std::atomic<std::int64_t> m_upstreamNs{ -1 };                              ///< Local receipt time of the report.

    /// Default-off configuration: let the assisted learner append, learn and score ACC windows without upstream
    /// lock. It never publishes: assisted output still requires lock, and a fit that learned without lock must be
    /// reset before it can predict.
    bool m_assistedDiagnostic{ false };

    /// The assisted fit learned from frames without upstream lock; prediction is refused until a model reset.
    std::atomic<bool> m_unlockedTraining{ false };

    /// Latest WFS-thread ACC window is complete and fresh, independently of upstream lock.
    bool m_windowValid{ false };

    /// Why the latest WFS-thread ACC window is not usable ("ok" when it is).
    std::atomic<const char *> m_windowReason{ "no_samples" };

    /// First failed outputValid() condition after the output was last qualified, and the ACC window reason then.
    /** Written by the output-thread qualification check; read by appLogic to explain a qualification fault. */
    mutable std::atomic<const char *> m_invalidReason{ "" }, m_invalidWindow{ "" };

    /// outputValid() last passed, so the next failure is recorded as the first one.
    mutable std::atomic<bool> m_wasValid{ false };

    std::uint64_t m_faultsLogged{ 0 }; ///< Output faults already logged; appLogic only.
    std::string   m_faultDetail;       ///< Latest fault's reason and condition, shown in safety; appLogic only.

    /// Shared WFS freshness policy, independent of ACC synchronization and the model's excitation rank.
    double                    m_inputTimeout{ .01 };
    std::atomic<std::int64_t> m_wfsReceipt{ -1 };
    std::atomic<bool>         m_inputValid{ false }, m_assistedValid{ false }, m_candidateValid{ false };

    /// Serialize numerical state and operator changes; the producer never takes this mutex.
    std::mutex                               m_modelMutex;
    std::unique_ptr<DDSPC::ShadowController> m_models;
    DDSPC::ShadowRecord                      m_pending;
    bool                                     m_havePending{ false };
    std::uint64_t                            m_wfsCounter{ 0 };
    std::array<DDSPC::Matrix, 2>             m_plans;

    /// DM slots per WFS frame, q: scheduler steps per cycle (four at 2 kHz give 8-kHz output).
    int m_slots{ 4 };

    /// ACC samples per WFS frame, r, set by the upstream synchronized sampling ratio.
    int m_ratio{ 4 };

    /// ACC base window in WFS frames; the window holds r*accel_history samples.
    int m_accelHistory{ 5 };

    /// Assisted weight on command variation within each future frame, relative to max(C^T C) like the command penalty.
    double m_ripple{ 10 };

    /// Size of the copied-slot ring: four frames of slots.
    int m_ring{ 16 };

    /// Brief plan/copied-command exchange with the producer worker, kept separate from all RLS and QR work.
    std::mutex m_commandMutex;

    /// Chronological modal command per slot for the latest decision; column j is slot m_planBase+1+j.
    DDSPC::Matrix m_slotPlan;

    /// Newest copied slot ordinal when m_slotPlan was decided; -1 indexes the plan from its first column.
    std::int64_t m_planBase{ -1 };

    /// Copied modal command of recent slot ordinals (column ordinal % m_ring); gaps hold the retained copy.
    DDSPC::Matrix m_appliedSlots;

    std::int64_t m_appliedOrdinal{ -1 }; ///< Newest copied slot ordinal; -1 until a copy in this scheduler epoch.

    std::int64_t m_appliedEpoch{ 0 }, m_appliedGeneration{ 0 }; ///< Scheduler incarnation of the ordinals.

    std::int64_t m_steps{ 4 }; ///< Scheduler steps per cycle from the latest record.

    std::int64_t m_decisionOrdinal{ -1 }; ///< m_appliedOrdinal at the previous accepted WFS decision.

    bool m_slotRestart{ false }; ///< Slot history restarted since the last decision; its snapshot ordinal is stale.

    std::uint64_t m_slotShifts{ 0 }; ///< Decisions whose interval contained other than zero or q new copies.

    DDSPC::Matrix m_copiedCommand;         ///< Newest copied modal command, used before slot ordinals are known.
    bool          m_commandKnown{ false }; ///< m_copiedCommand is finite and current.

    SyncoOutput m_output; ///< Bounded enable/zero/receipt state around the existing producer.

    std::thread       m_outputThread;             ///< Framework DM worker running the producer loop.
    bool              m_outputThreadInit{ true }; ///< threadStart synchronizer for priority and cpuset setup.
    pid_t             m_outputThreadID{ 0 };      ///< Worker thread ID, set by the worker for threadStart.
    pcf::IndiProperty m_outputThreadProp;         ///< Framework thread property (th-dm: pid and priority).
    int               m_outputThreadPrio{ 0 };    ///< Worker real-time priority; 0 keeps SCHED_OTHER.
    std::string       m_outputThreadCpuset;       ///< Worker cpuset; empty leaves placement unchanged.

    /// Bounded latency distribution over one INDI update interval, in 1-us bins.
    struct Latency
    {
        std::uint64_t                   m_count{ 0 }; ///< Samples.
        double                          m_sum{ 0 };   ///< Sum [us].
        double                          m_min{ 0 };   ///< Minimum [us].
        double                          m_max{ 0 };   ///< Maximum [us].
        std::array<std::uint32_t, 2048> m_bins{};     ///< Counts by whole microsecond; the last bin collects the rest.

        /// Add one sample [us].
        void add( double us /**< [in] Sample. */ );

        /// Mean [us], zero without samples.
        double mean() const;

        /// Upper bin edge of the given quantile [us], zero without samples.
        double quantile( double fraction /**< [in] Quantile in (0,1]. */ ) const;
    };

    /// Assisted multirate learner; owned by the WFS thread under m_modelMutex.
    std::unique_ptr<DDSPC::MultirateDDSPC> m_multirate;

    /// Assisted frame state handed to the DM worker.
    struct FastFrame
    {
        DDSPC::MultirateOrigin m_origin;           ///< Newest WFS origin.
        std::int64_t           m_base{ -1 };       ///< Newest copied ordinal at that WFS decision; -1 when unknown.
        DDSPC::VectorD         m_feedback;         ///< Added loPredCtrl feedback gain*y.
        double                 m_sigma{ 0 };       ///< Per-slot probe strength while exploring.
        bool                   m_predict{ false }; ///< Prediction requested and not faulted.
    };

    DDSPC::MultirateOrigin m_origin; ///< WFS-thread scratch origin.

    FastFrame     m_frame;             ///< Latest published frame, under m_commandMutex.
    std::uint64_t m_frameVersion{ 0 }; ///< Increments with every published frame.

    /// Triple-buffered gain sets: the WFS thread fills back, swaps it with middle under m_commandMutex; the worker
    /// swaps middle with front when fresh. Each side owns its buffer, so neither waits for the other's work.
    std::array<DDSPC::MultirateGains, 3> m_gainSets;
    int                                  m_gainBack{ 0 }, m_gainMiddle{ 1 }, m_gainFront{ 2 }; ///< Buffer roles.
    bool                                 m_gainFresh{ false }; ///< Middle holds a newer set than front.

    /// DM-worker-owned assisted state.
    FastFrame                        m_fastFrame;               ///< Worker copy of the latest frame.
    std::uint64_t                    m_fastVersion{ 0 };        ///< m_frameVersion of the copy.
    DDSPC::AdcCursor                 m_fastCursor;              ///< Worker ADC reader position.
    std::array<DDSPC::AdcSample, 64> m_fastSamples{};           ///< Newest ACC samples, indexed by counter modulo 64.
    std::uint64_t                    m_fastNewest{ 0 };         ///< Newest counter in m_fastSamples.
    std::uint64_t                    m_fastOldest{ 0 };         ///< Oldest counter of the current contiguous run.
    bool                             m_fastHaveSample{ false }; ///< m_fastSamples holds a sample.
    DDSPC::VectorD                   m_fastLast;    ///< Last rendered modal command, held when no plan applies.
    DDSPC::VectorD                   m_fastCommand; ///< Command workspace.
    DDSPC::Matrix                    m_fastFloat;   ///< Single-precision command for rendering.
    std::vector<double>              m_fastPrefix, m_fastExtras; ///< Executed slots and new ACC samples.
    bool                             m_fastClear{ false };       ///< Clear the held command (prediction re-enabled).
    std::default_random_engine       m_fastGenerator;            ///< Per-slot probe source of the assisted model.
    std::normal_distribution<double> m_fastDistribution{ 0, 1 };

    std::int64_t m_renderLeadNs{ 40000 }; ///< Assisted slots are calculated at cutoff minus this lead [ns].

    /// Assisted per-slot diagnostics, under m_commandMutex; latency distributions cover one INDI interval.
    struct FastStats
    {
        std::uint64_t m_predictions{ 0 };   ///< Slot calculations with a ready model.
        std::uint64_t m_slots{ 0 };         ///< Assisted slots rendered.
        std::uint64_t m_holds{ 0 };         ///< Slots without a current frame (late WFS), holding the last command.
        std::uint64_t m_lagging{ 0 };       ///< Slots with fewer new ACC samples than executed slots (k < p).
        std::uint64_t m_anomalies{ 0 };     ///< ACC counters outside the frame's [anchor, anchor+r) range.
        std::uint64_t m_mismatch{ 0 };      ///< Requests whose scheduler steps differ from dm_slots.
        std::vector<std::uint64_t> m_phase; ///< Slot calculations by ACC phase k.
        Latency                    m_compute, m_margin, m_wake, m_ageCalc, m_ageApply; ///< [us]
    };
    FastStats m_fastStats;

    Latency       m_learnUs;            ///< WFS-thread assisted frame work [us], under m_modelMutex.
    std::uint64_t m_learnOverruns{ 0 }; ///< Assisted frame work longer than the preceding WFS interval.
    std::int64_t  m_lastFrameNs{ -1 };  ///< Previous assisted WFS decision time.

    /// Counters at the previous INDI update, for rates; appLogic only.
    struct Rates
    {
        std::int64_t  m_ns{ -1 };
        std::uint64_t m_published{ 0 }, m_predictions{ 0 }, m_updates{ 0 }, m_refreshes{ 0 }, m_mismatch{ 0 };
    };
    Rates m_rates;

    /// Operator intent; learning OFF freezes both models while prediction can continue.
    bool                           m_learning{ false }, m_predicting{ false };
    std::atomic<DDSPC::ShadowMode> m_mode{ DDSPC::ShadowMode::Standard };
    DDSPC::realT                   m_gain{ .3f };

    /** \name Standard integrator - Data
     * Optional mfilt-equivalent output of standard mode: s = mult*s - gain*delta, where delta is the DDSPC command, as
     * the original loPredCtrl's commands were integrated by the main loop. See
     * agents/plans/2026-11/ddspc-standard-integrator.md.
     * @{ */
    bool          m_integrate{ false };        ///< Integrate standard commands; the fit is trained on the deltas.
    DDSPC::realT  m_intGain{ 1 };              ///< mfilt-equivalent gain applied to each delta.
    DDSPC::realT  m_intMult{ 1 };              ///< mfilt-equivalent multiplier of the previous state; below 1 leaks.
    DDSPC::realT  m_intLimit{ 0 };             ///< Per-mode absolute limit on the state [modal units].
    DDSPC::Matrix m_intState;                  ///< Integrator state, the held standard position; WFS thread.
    DDSPC::Matrix m_intPrevious;               ///< Mean copied position of the previous decision interval.
    bool          m_intPreviousKnown{ false }; ///< m_intPrevious is valid, so the next delta can be recovered.
    /// @}

    /// Original count, Gaussian strength, regularization exploration stages, consumed once per WFS decision.
    struct Stage
    {
        unsigned m_count{ 0 };                  ///< Remaining real WFS decisions.
        float    m_sigma{ 0 }, m_lambda{ 100 }; ///< Modal command strength and common command penalty.
    };
    std::vector<Stage>              m_exploration;
    std::default_random_engine      m_generator;
    std::normal_distribution<float> m_distribution{ 0, 1 };

    /// Public controls and compact diagnostics; names retain the operational loPredCtrl workflow.
    pcf::IndiProperty m_indiP_learning, m_indiP_predicting, m_indiP_learnStd, m_indiP_reset;
    pcf::IndiProperty m_indiP_mode, m_indiP_gain, m_indiP_exploration, m_indiP_sync;
    pcf::IndiProperty m_indiP_control, m_indiP_performance, m_indiP_safety, m_indiP_adc;

    /// Every assisted-mode qualification condition and the upstream values behind it.
    pcf::IndiProperty m_indiP_qualification;

    /// Operator recovery requests: re-zero the own DM channel, and return tip/tilt to the main loop.
    pcf::IndiProperty m_indiP_zeroDm, m_indiP_releaseTT;

    /// The last predict or learn_std request was refused; predict shows INDI_ALERT until a request is accepted.
    bool m_predictRejected{ false }; ///< Under m_indiMutex.

    /// Per-phase ddspc_performance element names (rmse_phase<k>, acc_phase<k>), owned for the INDI update vector.
    std::vector<std::string> m_phaseNames;

    /** \name Main-loop modal output - Data
     * @{ */
    /// parameters.own_shmim, as loPredCtrl: destroy the output at shutdown. Default false keeps it for consumers.
    bool m_modalOwnShmim{ false };

    std::vector<float> m_modalScratch;         ///< WFS-thread copy of the coherent input frame, edited before exchange.
    std::vector<float> m_modalFrame;           ///< Latest output frame, under m_modalMutex.
    timespec           m_modalAtime{};         ///< WFS acquisition time of m_modalFrame, under m_modalMutex.
    bool               m_modalFresh{ false };  ///< m_modalFrame is not yet in the stream, under m_modalMutex.
    std::mutex         m_modalMutex;           ///< Frame exchange between the WFS thread and the frameGrabber thread.
    std::uint64_t      m_modalCounter{ 0 };    ///< WFS counter of the latest output frame; WFS thread only.
    sem_t              m_modalSemaphore;       ///< Posted once per new frame, as loPredCtrl's m_smSemaphore.
    int                m_modalLock{ -1 };      ///< Exclusive `<stream>.producer` lock held while publishing.
    bool               m_modalLogged{ false }; ///< A refused attachment was logged; frameGrabber thread only.

    /// Prediction was accepted since the own-channel zero was last confirmed; written under m_indiMutex.
    /** Distinguishes removal of a DDSPC contribution, during which tip/tilt stays zeroed, from the startup zero. */
    std::atomic<bool> m_modalEngaged{ false };
    /// @}

    /// Load the existing calibration or selected finite masked cube planes; derive the request projection by QR.
    void loadCalibration();

    /// Allocate the independent pair and pre-sized workspaces without changing operator intent.
    void resetModels();

    /// Train only the selected mode's model: standard while standard is selected, the multirate learner otherwise.
    void applyTraining();

    /// Return whether the selected mode's model is ready.
    bool selectedReady() const;

    /// Freeze the causally available raw ACC history; never generate, interpolate, average or synchronize samples.
    bool acceleration( DDSPC::Matrix &features /**< [out] Newest pair first, r pairs per WFS history. */,
                       std::int64_t   now /**< [in] Frozen WFS decision cutoff. */ );

    /// Assemble the causal sensor vector from the existing reader's owned batch and continuity metadata.
    bool sensorHistory( const DDSPC::AdcBatch            &copied /**< [in] Coherent reader status and count. */,
                        std::span<const DDSPC::AdcSample> batch /**< [in] Samples available at decision time. */,
                        DDSPC::Matrix                    &features /**< [out] Newest-first X/Y pairs. */,
                        std::int64_t                      now /**< [in] Frozen WFS cutoff. */ );

    /// Copy the q slot commands applied since the previous decision; caller holds m_commandMutex.
    /** Returns the newest copied ordinal. applied is modes by m_slots, oldest slot first, or empty when unknown.
     * Fewer than q new copies (missed slots or idle output) are preceded by the retained previous copy.
     */
    std::int64_t appliedSlots( DDSPC::Matrix &applied /**< [out] Chronological copied modal commands. */ ) const;

    /// Return the standard fit's input for the interval just closed, and whether it is known.
    /** The input is the mean copied slot command, or with the integrator the delta that produced it,
     * -(p_k - mult*p_(k-1))/gain, so missed slots and clipping are seen as applied. Call once per decision so the
     * previous position stays current; an interval without a complete slot history makes the next delta unknown.
     */
    bool standardInput( const DDSPC::Matrix &applied /**< [in] Slot commands copied since the last decision. */,
                        DDSPC::Matrix       &input /**< [out] Mean position or integrator delta, when known. */ );

    /// Process one WFS decision, closing the preceding interval with the commands copied during it.
    int step( const DDSPC::Matrix &measurement /**< [in] Current calibrated WFS modes. */,
              std::uint64_t        counter /**< [in] Original WFS publication counter. */,
              const DDSPC::Matrix &features /**< [in] Individual raw ACC lift. */,
              bool                 assistedValid /**< [in] Upstream lock and complete causal sensor history. */,
              const DDSPC::Matrix &applied /**< [in] Slot commands copied since the last decision, or empty. */,
              std::int64_t         ordinal /**< [in] Newest copied slot ordinal at decision time, or -1. */,
              std::int64_t         now /**< [in] Current monotonic decision time. */ );

    /// Assisted WFS frame: append to the multirate learner, publish the origin, then learn and refresh the gains.
    /** Caller holds m_modelMutex. Standard learning is paused while assisted is selected; its histories continue. */
    int stepAssisted( const DDSPC::Matrix &measurement /**< [in] Current calibrated WFS modes. */,
                      std::uint64_t        counter /**< [in] Original WFS publication counter. */,
                      const DDSPC::Matrix &features /**< [in] ACC base window ending at m_adcAnchor. */,
                      bool                 assistedValid /**< [in] Upstream lock and complete causal sensor history. */,
                      const DDSPC::Matrix &applied /**< [in] Slot commands copied since the last decision. */,
                      std::int64_t         ordinal /**< [in] Newest copied slot ordinal at decision time, or -1. */,
                      std::int64_t         now /**< [in] Current monotonic decision time. */ );

    /// Check only current WFS/selected sensor input; bootstrap prediction may explore before a model is trained.
    bool outputValid() const;

    /// Render the modal command of an actual existing producer Request.
    bool outputCandidate( const synco::Request &request /**< [in] Exact slot being prepared. */,
                          const synco::Record  &schedule /**< [in] Coherent schedule, for steps per cycle. */,
                          std::span<float>      pixels /**< [out] Complete own auxiliary payload. */ );

    /// Calculate an assisted slot at cutoff minus the render lead from the newest ACC samples and executed slots.
    bool assistedCandidate( const synco::Request &request /**< [in] Exact slot being prepared. */,
                            const synco::Record  &schedule /**< [in] Coherent schedule, for steps per cycle. */,
                            std::span<float>      pixels /**< [out] Complete own auxiliary payload. */ );

    /// Read new ACC samples into the worker's ring; return the newest sample, or nullptr.
    const DDSPC::AdcSample *fastSamples( std::int64_t now /**< [in] Availability cutoff. */ );

    /// Insert one ACC sample into the worker's ring; a counter or segment discontinuity restarts the ring.
    void fastSample( const DDSPC::AdcSample &sample /**< [in] Decoded upstream sample. */ );

    /// Retain the exact transport-copied own contribution of a slot, separately from physical application.
    void copied( const synco::CommitRecord &commit /**< [in] Acknowledged commit identity. */,
                 std::span<const float>     pixels /**< [in] Matching acknowledged own payload. */ );

    /// Worker entry for MagAOXApp::threadStart; waits for priority/cpuset setup, then runs the producer loop.
    static void outputThreadStart( loPredCtrlAcc_operational *app /**< [in] Application owning the worker. */ );

    /// Request removal, join the worker and report whether the own zero was acknowledged.
    bool stopOutput();

    /// Validate the original exploration CSV grammar before replacing the running sequence.
    static std::vector<Stage> parseExploration( const std::string &text /**< [in] count,sigma,lambda triples. */ );

    /// Parse an entire finite INDI number; the library's stream extraction does not report conversion failures.
    static double number( const pcf::IndiElement &element /**< [in] Original wire value. */ );

    /// Reject mode/reset changes until output is off and its own zero has been acknowledged.
    bool removed() const;

    /// Turn prediction off, which schedules removal of the own-channel contribution; called under both INDI locks.
    void stopPrediction();

    /// Register operator controls without opening inputs or the auxiliary transport.
    int setupINDI();

    /// Validate the input vector layout on connection or source replacement.
    int allocate( const dev::shmimT &tag /**< [in] WFS monitor tag. */ );

    /// Run normal WFS-rate learning and prediction on the existing monitor thread.
    int processImage( void              *source /**< [in] Coherent FLOAT WFS modal payload. */,
                      const dev::shmimT &tag /**< [in] WFS monitor tag. */ );

    /// Reject a modal output that could feed back into the WFS input or zero other than modes 0 and 1.
    void validateModalOutput() const;

    /// Return whether DDSPC owns tip/tilt, so the main-loop output must carry zero for modes 0 and 1.
    /** True while prediction is requested, and after it stops until the own-channel zero is confirmed, so DDSPC and the
     * main loop never both correct tip/tilt. A startup zero alone (engaged false) never withholds main-loop tip/tilt.
     */
    static bool modalZeroed( const SyncoOutputStatus &output /**< [in] Own-channel output status. */,
                             bool engaged /**< [in] Prediction accepted since the last confirmed zero. */ );

    /// Zero tip/tilt in m_modalScratch while DDSPC owns it, exchange it for the frameGrabber and signal it; WFS thread.
    void publishModal( std::uint64_t   counter /**< [in] WFS counter of the copied frame. */,
                       const timespec &atime /**< [in] WFS acquisition time of the copied frame. */ );

    /// Log a refused modal-output attachment once and return -1, so the frameGrabber retries.
    int modalRefused( const std::string &reason /**< [in] Why the output stream cannot be used yet. */ );

    /** \name frameGrabber Interface
     * As loPredCtrl, publishing the latest modal frame each time the WFS thread posts m_modalSemaphore.
     * @{ */

    /// Take the exclusive output lock, then attach an idle existing stream of the same shape or size a new one.
    int configureAcquisition();

    /// Return 0: the output follows the WFS stream, so frameGrabber latency statistics are not configured.
    float fps();

    /// Nothing to start; frames come from the WFS thread.
    int startAcquisition();

    /// Wait up to 1 s for a new frame; 0 when one is ready, 1 on timeout.
    int acquireAndCheckValid();

    /// Copy the latest modal frame and its WFS acquisition time into the stream.
    int loadImageIntoStream( void *dest /**< [out] Output stream buffer. */ );

    /// Nothing to reconfigure; configureAcquisition() re-reads the input shape.
    int reconfig();
    ///@}

    /** \name telemeter Interface
     * @{ */

    /// Record frameGrabber timing telemetry when due.
    int checkRecordTimes();

    /// Record frameGrabber timing telemetry.
    int recordTelem( const telem_fgtimings * /**< [in] Telemetry type tag. */ );
    ///@}

  public:
    /// Set operational model defaults before reading configuration.
    loPredCtrlAcc_operational();

    /// Stop callbacks while the application's command/model state still exists.
    ~loPredCtrlAcc_operational() noexcept;

    /// Register the existing base interfaces and operational configuration.
    virtual void setupConfig();

    /// Validate configuration and calibration; missing ACC configuration leaves standard control available.
    virtual void loadConfig();

    /// Load the base interfaces through their integer-returning configuration macros.
    int loadConfigImpl();

    /// Register controls, start input monitoring, and acquire only the existing auxiliary producer.
    virtual int appStartup();

    /// Publish model/transport state without placing INDI activity in the WFS callback.
    virtual int appLogic();

    /// Remove only this app's contribution before closing producer and input resources.
    virtual int appShutdown();

    /// Set learning for the selected mode's model; the other model stays frozen.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_learning );

    /// Enable prediction/bootstrap exploration, or request acknowledged own-channel zero.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_predicting );

    /// Run the original three-stage standard learning sequence.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_learnStd );

    /// Reset both models after confirmed command removal.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_reset );

    /// Stop prediction and write a fresh zero to the own DM channel, even if its removal was already confirmed.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_zeroDm );

    /// Stop prediction and pass main-loop tip/tilt through at once, without waiting for the zero to be confirmed.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_releaseTT );

    /// Select the active independent fit; learning follows the selection.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_mode );

    /// Set the additive current-residual feedback gain used by loPredCtrl.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_gain );

    /// Install an original-format explicit exploration sequence.
    INDI_NEWCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_exploration );

    /// Receive the existing upstream synchronization state; it never gates standard DDSPC.
    INDI_SETCALLBACK_DECL( loPredCtrlAcc_operational, m_indiP_sync );
};

inline loPredCtrlAcc_operational::loPredCtrlAcc_operational() : MagAOXApp( MAGAOX_CURRENT_SHA1, MAGAOX_REPO_MODIFIED )
{
    m_settings.m_modes                = 2;
    m_settings.m_prior                = 1e-5f;
    m_settings.m_accelHistory         = 1;
    m_settings.m_useQrd               = true;
    m_settings.m_assistedCommandSlots = 1; // The shadow assisted fit is unused; assisted mode is MultirateDDSPC.
    m_settings.m_scaledPenalty = m_settings.m_externalOrigins = m_settings.m_freezeQualification = true;
    m_settings.m_requireFullRank                                                                 = false;
    m_calibration.m_indices                                                                      = { 0, 1 };
}

inline loPredCtrlAcc_operational::~loPredCtrlAcc_operational() noexcept
{
    m_shutdown = true;
    stopOutput();
    frameGrabberT::appShutdown(); // Join the publisher before its frame storage is destroyed.
    shmimMonitorT::appShutdown();
    if( m_modalLock >= 0 )
        close( m_modalLock );
}

inline bool loPredCtrlAcc_operational::stopOutput()
{
    m_output.stop();
    if( m_outputThread.joinable() )
        m_outputThread.join();
    return m_output.status().m_zeroConfirmed;
}

inline void loPredCtrlAcc_operational::outputThreadStart( loPredCtrlAcc_operational *app )
{
    app->m_outputThreadID = syscall( SYS_gettid );
    // Wait for threadStart to finish priority and cpuset placement, as dmSpeckle and accelSim do.
    while( app->m_outputThreadInit && !app->m_shutdown )
        std::this_thread::sleep_for( std::chrono::milliseconds( 10 ) );
    app->m_output.work(); // Runs until stopOutput(), including the final own-channel zero.
}

inline void loPredCtrlAcc_operational::setupConfig()
{
    SHMIMMONITOR_SETUP_CONFIG( config );
    DMSYNCOPATIONPRODUCER_SETUP_CONFIG( config );
    FRAMEGRABBER_SETUP_CONFIG( config );
    TELEMETER_SETUP_CONFIG( config );
    auto add =
        [this]( const std::string &section, const std::string &key, const std::string &type, const std::string &help )
    { config.add( section + "." + key, "", section + "." + key, argType::Required, section, key, false, type, help ); };
    for( const auto &key : { "gain", "regularization", "gamma", "covariance", "input_timeout" } )
        add( "parameters", key, "float", "Operational loPredCtrl parameter; input_timeout is in seconds." );
    add( "parameters",
         "assisted_diagnostic",
         "bool",
         "Assisted learner/scoring without upstream lock; never publishes; default false." );
    add( "parameters",
         "ripple",
         "float",
         "Assisted within-frame command-variation weight, scaled like regularization (10)." );
    for( const auto &key : { "num_modes", "history", "future", "accel_history", "modal_index0", "modal_index1" } )
        add( "parameters", key, "int", "Model dimensions or WFS modal indices; accel_history counts WFS periods." );
    add( "parameters", "dm_slots", "int", "DM slots per WFS frame: the scheduler's steps per cycle (default 4)." );
    add( "parameters", "acc_ratio", "int", "Upstream ACC samples per WFS frame (default 4)." );
    add( "parameters", "qrd", "bool", "Use QR-RLS for both independent models." );
    add( "integrator",
         "enabled",
         "bool",
         "Integrate standard commands as mfilt does, s = mult*s - gain*delta, and train on the deltas (default "
         "false)." );
    add( "integrator", "gain", "float", "mfilt-equivalent gain on each DDSPC delta, in (0,10] (default 1)." );
    add( "integrator", "mult", "float", "mfilt-equivalent multiplier of the previous state, in (0,1] (default 1)." );
    add( "integrator", "limit", "float", "Per-mode absolute state limit [modal units]; required when enabled." );
    add( "parameters",
         "own_shmim",
         "bool",
         "Destroy the modal output stream at shutdown, as loPredCtrl (default false: keep it for consumers)." );
    for( const auto &key : { "channelName", "modeCube", "mask", "calibration" } )
        add( "dm", key, "string", "Existing own auxiliary/calibration; calibration overrides modeCube/mask." );
    add( "dm", "threadPrio", "int", "DM worker real-time priority [0,99], as dmSpeckle; 0 (default) is SCHED_OTHER." );
    add( "dm", "cpuset", "string", "DM worker cpuset; empty (default) leaves placement unchanged." );
    add( "dm",
         "renderLeadUs",
         "float",
         "Assisted slots are calculated this long before their cutoff [us] (default 40)." );
    add( "adc", "enabled", "bool", "Use upstream synchronized ADC inputs for the assisted model." );
    for( const auto &key : { "shmimName", "profile", "sync_device", "sync_property" } )
        add( "adc", key, "string", "Existing ADC source or existing upstream lock diagnostic." );
    for( const auto &key : { "min_code", "max_code" } )
        add( "adc", key, "int", "Raw ADC encoding bounds." );
    add( "adc", "capacity", "int", "Existing ADC reader ring capacity." );
    for( const auto &key : { "max_age_us", "max_interval_us", "clock_tolerance_us" } )
        add( "adc", key, "float", "Existing ADC reader validity interval in microseconds." );
    add( "adc", "reject_rails", "bool", "Reject ADC rail samples." );
    for( const auto &key : { "zero", "volts_per_count", "sensitivity", "polarity" } )
        for( int axis = 0; axis < 2; ++axis )
            add( "adc", std::string( key ) + std::to_string( axis ), "float", "Per-axis upstream ADC calibration." );
}

inline void loPredCtrlAcc_operational::loadConfig()
{
    if( loadConfigImpl() < 0 )
        m_shutdown = true;
}

inline int loPredCtrlAcc_operational::loadConfigImpl()
{
    try
    {
        SHMIMMONITOR_LOAD_CONFIG( config );
        DMSYNCOPATIONPRODUCER_LOAD_CONFIG( config );
        config( m_modalOwnShmim, "parameters.own_shmim" );
        frameGrabberT::m_ownShmim = m_modalOwnShmim;
        FRAMEGRABBER_LOAD_CONFIG( config );
        TELEMETER_LOAD_CONFIG( config );
        config( m_gain, "parameters.gain" );
        config( m_settings.m_lambda, "parameters.regularization" );
        config( m_settings.m_gamma, "parameters.gamma" );
        config( m_settings.m_prior, "parameters.covariance" );
        config( m_settings.m_modes, "parameters.num_modes" );
        config( m_settings.m_history, "parameters.history" );
        config( m_settings.m_future, "parameters.future" );
        config( m_settings.m_useQrd, "parameters.qrd" );
        config( m_inputTimeout, "parameters.input_timeout" );
        config( m_accelHistory, "parameters.accel_history" );
        config( m_slots, "parameters.dm_slots" );
        config( m_ratio, "parameters.acc_ratio" );
        config( m_ripple, "parameters.ripple" );
        config( m_assistedDiagnostic, "parameters.assisted_diagnostic" );
        double renderLeadUs = m_renderLeadNs / 1000.;
        config( renderLeadUs, "dm.renderLeadUs" );
        config( m_calibration.m_indices[0], "parameters.modal_index0" );
        config( m_calibration.m_indices[1], "parameters.modal_index1" );
        config( m_channel, "dm.channelName" );
        config( m_modeCube, "dm.modeCube" );
        config( m_mask, "dm.mask" );
        config( m_calibrationFile, "dm.calibration" );
        config( m_outputThreadPrio, "dm.threadPrio" );
        config( m_outputThreadCpuset, "dm.cpuset" );
        if( m_channel.size() <= 4 || !m_channel.ends_with( "_aux" ) || m_outputThreadPrio < 0 ||
            m_outputThreadPrio > 99 || m_syncopationTiming.empty() || shmimMonitorT::m_shmimName.empty() ||
            !std::isfinite( m_gain ) || std::abs( m_gain ) > 10 || !std::isfinite( m_inputTimeout ) ||
            m_inputTimeout <= 0 || m_inputTimeout > 1 || m_settings.m_modes < 1 || m_settings.m_modes > 2 ||
            m_accelHistory < 1 || m_accelHistory > 16 || m_slots < 1 || m_slots > 16 || m_ratio < 1 || m_ratio > 16 ||
            !std::isfinite( renderLeadUs ) || renderLeadUs < 0 || renderLeadUs > 1000 || !std::isfinite( m_ripple ) ||
            m_ripple < 0 )
            throw std::invalid_argument( "Invalid operational WFS, producer, gain or model configuration" );

        config( m_integrate, "integrator.enabled" );
        config( m_intGain, "integrator.gain" );
        config( m_intMult, "integrator.mult" );
        config( m_intLimit, "integrator.limit" );

        if( m_integrate &&
            ( !std::isfinite( m_intGain ) || m_intGain <= 0 || m_intGain > 10 || !std::isfinite( m_intMult ) ||
              m_intMult <= 0 || m_intMult > 1 || !std::isfinite( m_intLimit ) || m_intLimit <= 0 ) )
        {
            throw std::invalid_argument( "Invalid integrator gain, mult or limit" );
        }

        if( m_integrate )
        {
            log<text_log>( "Standard integrator enabled: gain=" + std::to_string( m_intGain ) +
                               " mult=" + std::to_string( m_intMult ) + " limit=" + std::to_string( m_intLimit ),
                           logPrio::LOG_NOTICE );
        }

        m_renderLeadNs             = std::llround( renderLeadUs * 1000 );
        m_ring                     = 4 * m_slots;
        m_settings.m_accelChannels = 2 * m_ratio * m_accelHistory;
        loadCalibration();
        validateModalOutput(); // After the calibration, which can replace the modal indices.
        resetModels();
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    try
    {
        config( m_accelEnabled, "adc.enabled" );
        if( !m_accelEnabled )
            return 0;
        config( m_adcConfig.m_capacity, "adc.capacity" );
        auto interval = [this]( const char *key, std::int64_t &nanoseconds )
        {
            double microseconds = nanoseconds / 1000.;
            config( microseconds, key );
            if( !std::isfinite( microseconds ) || microseconds < .001 || microseconds > 1e9 )
                throw std::invalid_argument( std::string( "Invalid ADC interval: " ) + key );
            nanoseconds = std::llround( microseconds * 1000 );
        };
        interval( "adc.max_age_us", m_adcConfig.m_maxAgeNs );
        interval( "adc.max_interval_us", m_adcConfig.m_maxIntervalNs );
        interval( "adc.clock_tolerance_us", m_adcConfig.m_clockToleranceNs );
        config( m_adcConfig.m_stream, "adc.shmimName" );
        config( m_adcConfig.m_profile, "adc.profile" );
        config( m_syncDevice, "adc.sync_device" );
        config( m_syncProperty, "adc.sync_property" );
        config( m_adcConfig.m_minCode, "adc.min_code" );
        config( m_adcConfig.m_maxCode, "adc.max_code" );
        config( m_adcConfig.m_rejectRails, "adc.reject_rails" );
        for( unsigned axis = 0; axis < 2; ++axis )
        {
            const auto suffix = std::to_string( axis );
            config( m_adcConfig.m_zero[axis], "adc.zero" + suffix );
            config( m_adcConfig.m_voltsPerCount[axis], "adc.volts_per_count" + suffix );
            config( m_adcConfig.m_sensitivity[axis], "adc.sensitivity" + suffix );
            config( m_adcConfig.m_polarity[axis], "adc.polarity" + suffix );
        }
        DDSPC::validate_adc_config( m_adcConfig );
    }
    catch( const std::exception &error )
    {
        m_accelEnabled = false;
        log<software_error>( { __FILE__, __LINE__, std::string( "ACC assistance unavailable: " ) + error.what() } );
    }
    return 0;
}

inline void loPredCtrlAcc_operational::loadCalibration()
{
    if( !m_calibrationFile.empty() )
        m_calibration = DDSPC::load_native_calibration( m_calibrationFile );
    else
    {
        if( m_modeCube.empty() )
            throw std::invalid_argument( "Supply the operational modal-to-DM modeCube" );
        mx::fits::fitsFile<float>     file;
        mx::improc::eigenCube<float>  cube;
        mx::improc::eigenImage<float> mask;
        if( file.read( cube, m_modeCube ) != mx::error_t::noerror || cube.rows() < 1 || cube.cols() < 1 ||
            cube.planes() < 1 )
            throw std::invalid_argument( "Could not read DM mode cube: " + m_modeCube );
        if( m_mask.empty() )
            mask = mx::improc::eigenImage<float>::Ones( cube.rows(), cube.cols() );
        else if( file.read( mask, m_mask ) != mx::error_t::noerror )
            throw std::invalid_argument( "Could not read DM mask: " + m_mask );
        if( mask.rows() != cube.rows() || mask.cols() != cube.cols() || !mask.allFinite() ||
            cube.rows() * cube.cols() > 4096 )
            throw std::invalid_argument( "Incompatible DM modal cube/mask" );
        m_calWidth  = cube.rows();
        m_calHeight = cube.cols();
        m_calibration.m_shapes.resize( mask.size(), m_settings.m_modes );
        for( int mode = 0; mode < m_settings.m_modes; ++mode )
        {
            const auto index = m_calibration.m_indices[mode];
            if( index >= std::size_t( cube.planes() ) || !cube.image( index ).allFinite() )
                throw std::invalid_argument( "Missing or nonfinite calibrated DM mode" );
            for( int pixel = 0; pixel < mask.size(); ++pixel )
                m_calibration.m_shapes( pixel, mode ) = mask.data()[pixel] > 0 ? cube.image( index ).data()[pixel] : 0;
        }
        Eigen::CompleteOrthogonalDecomposition<DDSPC::Matrix> qr( m_calibration.m_shapes );
        if( qr.rank() != m_settings.m_modes )
            throw std::invalid_argument( "DM modes have deficient rank on the supplied mask" );
        m_calibration.m_projection = qr.pseudoInverse();
        m_calibration.m_layout     = 1;
    }
    const auto &shapes = m_calibration.m_shapes;
    if( shapes.rows() <= 0 || shapes.cols() != m_settings.m_modes || !shapes.allFinite() ||
        m_calibration.m_projection.cols() != shapes.rows() || m_calibration.m_projection.rows() != m_settings.m_modes ||
        !m_calibration.m_projection.allFinite() )
        throw std::invalid_argument( "Missing complete calibrated DM synthesis/projection" );
    m_copiedCommand = DDSPC::Matrix::Zero( m_settings.m_modes, 1 );
    m_appliedSlots  = DDSPC::Matrix::Zero( m_settings.m_modes, m_ring );
}

inline void loPredCtrlAcc_operational::resetModels()
{
    m_models = std::make_unique<DDSPC::ShadowController>( m_settings );
    DDSPC::MultirateConfig multirate;
    multirate.m_modes        = m_settings.m_modes;
    multirate.m_history      = m_settings.m_history;
    multirate.m_future       = m_settings.m_future;
    multirate.m_slots        = m_slots;
    multirate.m_ratio        = m_ratio;
    multirate.m_accelHistory = std::max( 1, m_settings.m_accelChannels / ( 2 * m_ratio ) );
    multirate.m_gamma        = m_settings.m_gamma;
    multirate.m_prior        = m_settings.m_prior;
    multirate.m_lambda       = m_settings.m_lambda;
    multirate.m_ripple       = m_ripple;
    m_multirate              = std::make_unique<DDSPC::MultirateDDSPC>( multirate );
    m_multirate->allocate( m_origin );
    m_learnUs          = {};
    m_learnOverruns    = 0;
    m_unlockedTraining = false;
    m_lastFrameNs      = -1;
    applyTraining();
    m_plans[0] = DDSPC::Matrix::Zero( m_settings.m_future * m_settings.m_modes, 1 );
    m_plans[1] = DDSPC::Matrix::Zero( m_settings.m_future * m_settings.m_modes * m_settings.m_assistedCommandSlots, 1 );
    m_pending  = {};
    m_havePending      = false;
    m_wfsCounter       = 0;
    m_intState         = DDSPC::Matrix::Zero( m_settings.m_modes, 1 );
    m_intPreviousKnown = false;
    std::lock_guard<std::mutex> lock( m_commandMutex );
    m_slotPlan        = DDSPC::Matrix::Zero( m_settings.m_modes, m_settings.m_future * m_slots );
    m_planBase        = -1;
    m_decisionOrdinal = -1;
    for( auto &gains : m_gainSets )
        m_multirate->allocate( gains );
    m_gainFresh = false;
    for( auto *frame : { &m_frame, &m_fastFrame } )
    {
        m_multirate->allocate( frame->m_origin );
        frame->m_base     = -1;
        frame->m_feedback = DDSPC::VectorD::Zero( m_settings.m_modes );
        frame->m_sigma    = 0;
        frame->m_predict  = false;
    }
    m_frameVersion = m_fastVersion = 0;
    m_fastLast                     = DDSPC::VectorD::Zero( m_settings.m_modes );
    m_fastCommand                  = DDSPC::VectorD::Zero( m_settings.m_modes );
    m_fastFloat                    = DDSPC::Matrix::Zero( m_settings.m_modes, 1 );
    m_fastPrefix.assign( m_slots * m_settings.m_modes, 0 );
    m_fastExtras.assign( 2 * m_ratio, 0 );
    m_fastStats = {};
    m_fastStats.m_phase.assign( m_ratio, 0 );
}

inline void loPredCtrlAcc_operational::applyTraining()
{
    const bool assisted = m_mode.load() == DDSPC::ShadowMode::AccelAssisted;
    m_models->training( DDSPC::ShadowMode::Standard, m_learning && !assisted );
    m_models->training( DDSPC::ShadowMode::AccelAssisted, false );
    m_multirate->training( m_learning && assisted );
}

inline bool loPredCtrlAcc_operational::selectedReady() const
{
    return m_mode.load() == DDSPC::ShadowMode::Standard ? m_models->status( DDSPC::ShadowMode::Standard ).m_ready
                                                        : m_multirate->status().m_ready;
}

inline void loPredCtrlAcc_operational::Latency::add( double us )
{
    m_min = m_count ? std::min( m_min, us ) : us;
    m_max = m_count ? std::max( m_max, us ) : us;
    ++m_count;
    m_sum += us;
    ++m_bins[std::size_t( std::clamp( us, 0.0, double( m_bins.size() - 1 ) ) )];
}

inline double loPredCtrlAcc_operational::Latency::mean() const
{
    return m_count ? m_sum / m_count : 0;
}

inline double loPredCtrlAcc_operational::Latency::quantile( double fraction ) const
{
    std::uint64_t seen = 0;
    for( std::size_t bin = 0; bin < m_bins.size(); ++bin )
        if( ( seen += m_bins[bin] ) >= fraction * m_count && m_count )
            return std::min( double( bin + 1 ), m_max );
    return m_max;
}

inline bool loPredCtrlAcc_operational::acceleration( DDSPC::Matrix &features, std::int64_t now )
{
    m_windowValid = false;
    if( !m_adc )
    {
        m_windowReason.store( "adc_disabled" );
        return false;
    }
    std::array<DDSPC::AdcSample, 128> batch;
    const auto                        copied = m_adc->copy_after( m_adcCursor, batch, now );
    m_adcCursor                              = copied.m_next;
    return sensorHistory( copied, batch, features, now );
}

inline bool loPredCtrlAcc_operational::sensorHistory( const DDSPC::AdcBatch            &copied,
                                                      std::span<const DDSPC::AdcSample> batch,
                                                      DDSPC::Matrix                    &features,
                                                      std::int64_t                      now )
{
    const bool locked = m_syncLocked.load();
    m_windowValid     = false;
    auto reject       = [this]( const char *reason )
    {
        m_windowReason.store( reason );
        return false;
    };
    if( copied.m_count > batch.size() || !copied.m_status.m_valid || ( !locked && !m_assistedDiagnostic ) )
    {
        m_adcCount = m_adcNext = 0;
        return reject( !copied.m_status.m_valid || copied.m_count > batch.size() ? "adc_stream_invalid"
                                                                                 : "awaiting_upstream_lock" );
    }
    if( copied.m_discontinuity || copied.m_more )
        m_adcCount = m_adcNext = 0;
    for( unsigned n = 0; n < copied.m_count; ++n )
    {
        m_adcHistory[m_adcNext] = batch[n];
        m_adcNext               = ( m_adcNext + 1 ) % m_adcHistory.size();
        m_adcCount              = std::min<unsigned>( m_adcCount + 1, m_adcHistory.size() );
    }
    const unsigned count = m_settings.m_accelChannels / 2;
    if( !copied.m_count )
        return reject( "no_new_sample" );
    if( copied.m_more )
        return reject( "reader_backlog" );
    if( m_adcCount < count )
        return reject( "history_filling" );
    const auto &newest = m_adcHistory[( m_adcNext + m_adcHistory.size() - 1 ) % m_adcHistory.size()];
    if( now < newest.m_receivedNs || now - newest.m_acquiredNs > m_adcConfig.m_maxAgeNs )
        return reject( "stale_sample" );
    features.resize( m_settings.m_accelChannels, 1 );
    m_adcAnchor  = newest.m_frame.m_counter;
    m_adcSegment = ( newest.m_generation << 32 ) ^ newest.m_segment;
    for( unsigned n = 0; n < count; ++n )
    {
        const auto &sample = m_adcHistory[( m_adcNext + m_adcHistory.size() - n - 1 ) % m_adcHistory.size()];
        if( sample.m_frame.m_counter + n != newest.m_frame.m_counter || sample.m_segment != newest.m_segment ||
            sample.m_generation != newest.m_generation || sample.m_receivedNs > now )
            return reject( "counter_gap" );
        features( 2 * n )     = sample.m_acceleration[0];
        features( 2 * n + 1 ) = sample.m_acceleration[1];
    }
    if( !features.allFinite() )
        return reject( "nonfinite_sample" );
    m_windowValid = true;
    m_windowReason.store( locked ? "ok" : "ok_unlocked_diagnostic" );
    return locked; // Assisted control and output always require the upstream lock.
}

inline std::int64_t loPredCtrlAcc_operational::appliedSlots( DDSPC::Matrix &applied ) const
{
    applied.resize( m_settings.m_modes, 0 );
    if( m_appliedOrdinal >= 0 && m_decisionOrdinal >= 0 && m_appliedOrdinal >= m_decisionOrdinal )
    {
        applied.resize( m_settings.m_modes, m_slots );
        for( int j = 0; j < m_slots; ++j )
            applied.col( j ) =
                m_appliedSlots.col( std::max( m_appliedOrdinal - m_slots + 1 + j, m_decisionOrdinal ) % m_ring );
    }
    else if( m_commandKnown )
        applied = m_copiedCommand.replicate( 1, m_slots ); // No slot history yet: the retained copy is held.
    return m_appliedOrdinal;
}

inline bool loPredCtrlAcc_operational::standardInput( const DDSPC::Matrix &applied, DDSPC::Matrix &input )
{
    if( applied.rows() != m_settings.m_modes || applied.cols() != m_slots )
    {
        m_intPreviousKnown = false;
        return false;
    }

    // The standard single-rate input is the frame's mean slot command, equal to the held command in standard mode.
    const DDSPC::Matrix position = applied.rowwise().mean();
    if( !m_integrate )
    {
        input = position;
        return true;
    }

    const bool known = m_intPreviousKnown;
    if( known )
    {
        input = -( position - m_intMult * m_intPrevious ) / m_intGain;
    }

    m_intPrevious      = position;
    m_intPreviousKnown = true;
    return known;
}

inline int loPredCtrlAcc_operational::step( const DDSPC::Matrix &measurement,
                                            std::uint64_t        counter,
                                            const DDSPC::Matrix &features,
                                            bool                 assistedValid,
                                            const DDSPC::Matrix &applied,
                                            std::int64_t         ordinal,
                                            std::int64_t         now )
{
    std::lock_guard<std::mutex> lock( m_modelMutex );
    if( !m_models || measurement.rows() != m_settings.m_modes || measurement.cols() != 1 || !measurement.allFinite() ||
        !counter )
    {
        m_inputValid.store( false );
        return 0;
    }
    if( counter == m_wfsCounter )
        return 0;
    if( m_wfsCounter && counter != m_wfsCounter + 1 )
    {
        m_models->invalidate( {} );
        m_havePending = false;
    }

    // Every decision, so the integrator's previous position stays current across gaps and mode changes.
    DDSPC::Matrix input;
    m_pending.m_appliedValid = standardInput( applied, input );
    if( m_pending.m_appliedValid )
    {
        m_pending.m_appliedCommand = input;
    }

    const auto selected = m_mode.load();
    if( selected == DDSPC::ShadowMode::AccelAssisted )
        return stepAssisted( measurement, counter, features, assistedValid, applied, ordinal, now );
    const bool probing = m_predicting && m_learning && !m_exploration.empty() &&
                         ( selected == DDSPC::ShadowMode::Standard || assistedValid );
    if( probing )
        m_models->regularization( m_exploration.front().m_lambda );
    const int modes = m_settings.m_modes, frames = m_settings.m_future;
    if( m_havePending )
    {
        // The standard input was set by standardInput(); the shadow assisted bank keeps the individual slots.
        const bool known                 = applied.rows() == modes && applied.cols() == m_slots;
        m_pending.m_assistedAppliedValid = known;
        if( known )
        {
            m_pending.m_assistedAppliedCommand.resize( modes * m_slots, 1 );
            for( int j = 0; j < m_slots; ++j ) // Newest slot first, as ShadowRecord requires.
                m_pending.m_assistedAppliedCommand.middleRows( ( m_slots - 1 - j ) * modes, modes ) = applied.col( j );
        }
        m_models->process( m_pending );
    }
    // Column m_slots*f+j is slot j of frame f after this decision; plans are newest-first frames (and slots).
    bool          selectedReady = false;
    DDSPC::Matrix plan          = ( m_gain * measurement ).replicate( 1, frames * m_slots );
    for( unsigned model = 0; model < 2; ++model )
    {
        const auto    mode = model ? DDSPC::ShadowMode::AccelAssisted : DDSPC::ShadowMode::Standard;
        DDSPC::Matrix empty;
        const bool    ready =
            ( !model || assistedValid ) &&
            m_models->predict_next( mode, counter, measurement, model ? features : empty, m_plans[model] );
        if( mode == selected )
            selectedReady = ready;
        if( m_predicting && mode == selected && ready )
            for( int f = 0; f < frames; ++f )
                for( int j = 0; j < m_slots; ++j )
                    plan.col( m_slots * f + j ) +=
                        model
                            ? m_plans[1].middleRows( ( ( frames - 1 - f ) * m_slots + m_slots - 1 - j ) * modes, modes )
                            : m_plans[0].middleRows( ( frames - 1 - f ) * modes, modes );
    }
    const bool numericalFault = m_models->status( selected ).m_reason == DDSPC::ShadowReason::NumericalFault;
    if( probing && !numericalFault )
    {
        // Standard probes are held for the frame, as in loPredCtrl; assisted probes excite each slot independently.
        auto &stage = m_exploration.front();
        for( int mode = 0; mode < modes; ++mode )
        {
            float draw = stage.m_sigma * m_distribution( m_generator );
            for( int j = 0; j < m_slots; ++j )
            {
                if( j && selected == DDSPC::ShadowMode::AccelAssisted )
                    draw = stage.m_sigma * m_distribution( m_generator );
                plan( mode, j ) += draw;
            }
        }
        if( !--stage.m_count )
            m_exploration.erase( m_exploration.begin() );
    }
    if( !m_predicting || numericalFault )
    {
        plan.setZero();
        m_intState.setZero(); // Prediction always restarts from the confirmed own-channel zero.
    }
    else if( m_integrate && plan.allFinite() )
    {
        // mfilt: the first column is this frame's delta; the state is held on every slot until the next decision.
        m_intState =
            ( m_intMult * m_intState - m_intGain * plan.col( 0 ) ).cwiseMax( -m_intLimit ).cwiseMin( m_intLimit );
        plan = m_intState.replicate( 1, plan.cols() );
    }

    const bool finite = plan.allFinite();
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        if( m_slotRestart ) // The ordinal was read before a scheduler reset; re-anchor at the next decision.
            ordinal = -1;
        else if( ordinal >= 0 && m_decisionOrdinal >= 0 && ordinal != m_decisionOrdinal &&
                 ordinal - m_decisionOrdinal != m_slots )
            ++m_slotShifts;
        m_slotRestart     = false;
        m_decisionOrdinal = ordinal;
        if( finite )
        {
            m_slotPlan = plan;
            m_planBase = ordinal;
        }
        m_assistedValid.store( assistedValid );
        m_candidateValid.store( finite && !numericalFault &&
                                ( !m_predicting || selectedReady || probing || ( m_learning && m_gain != 0 ) ) );
        m_wfsReceipt.store( now );
        m_inputValid.store( true );
    }
    m_pending                = {};
    m_pending.m_index        = counter;
    m_pending.m_residual     = measurement;
    m_pending.m_acceleration = features;
    m_pending.m_wfsValid = m_pending.m_timingValid = true;
    m_pending.m_accelValid                         = assistedValid;
    m_havePending                                  = true;
    m_wfsCounter                                   = counter;
    return 0;
}

inline int loPredCtrlAcc_operational::stepAssisted( const DDSPC::Matrix &measurement,
                                                    std::uint64_t        counter,
                                                    const DDSPC::Matrix &features,
                                                    bool                 assistedValid,
                                                    const DDSPC::Matrix &applied,
                                                    std::int64_t         ordinal,
                                                    std::int64_t         now )
{
    const int  modes   = m_settings.m_modes;
    const bool probing = m_predicting && m_learning && !m_exploration.empty() && assistedValid;
    if( probing )
        m_multirate->regularization( m_exploration.front().m_lambda );
    // Keep the frozen standard fit's histories continuous so it resumes without a gap when standard is reselected.
    if( m_havePending ) // step() set the standard input, in the same units as the standard fit was trained on.
    {
        m_pending.m_assistedAppliedValid = false;
        m_pending.m_accelValid           = false;
        m_models->process( m_pending );
    }
    // Frame i: y_i, the q slots copied over [i-1, i) and the ACC base window ending at the anchor c_i.
    const DDSPC::VectorD y    = measurement.cast<double>();
    const DDSPC::MatrixD lift = applied.rows() == modes && applied.cols() == m_slots
                                    ? DDSPC::MatrixD( applied.cast<double>() )
                                    : DDSPC::MatrixD();
    // The default-off diagnostic mode lets the learner use complete windows without upstream lock. A fit that learns
    // from them is barred from prediction until reset; output still requires lock (outputValid).
    const bool diagnostic = !assistedValid && m_assistedDiagnostic && m_windowValid;
    if( diagnostic && m_multirate->status().m_training )
        m_unlockedTraining = true;
    const DDSPC::VectorD window =
        assistedValid || diagnostic ? DDSPC::VectorD( features.cast<double>() ) : DDSPC::VectorD();
    m_multirate->append( counter, y, lift, window, m_adcAnchor, m_adcSegment, m_origin );
    const bool ready = m_multirate->status().m_ready;
    const bool fault = !y.allFinite();
    { //mutex scope
        // Publish the origin before learning, so the frame's first slots already use y_i.
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        if( m_slotRestart )
            ordinal = -1;
        else if( ordinal >= 0 && m_decisionOrdinal >= 0 && ordinal != m_decisionOrdinal &&
                 ordinal - m_decisionOrdinal != m_slots )
            ++m_slotShifts;
        m_slotRestart     = false;
        m_decisionOrdinal = ordinal;
        std::swap( m_frame.m_origin.m_state, m_origin.m_state );
        m_frame.m_origin.m_index   = m_origin.m_index;
        m_frame.m_origin.m_anchor  = m_origin.m_anchor;
        m_frame.m_origin.m_segment = m_origin.m_segment;
        m_frame.m_origin.m_valid   = m_origin.m_valid;
        m_frame.m_base             = ordinal;
        m_frame.m_feedback         = m_gain * y;
        m_frame.m_sigma            = probing ? m_exploration.front().m_sigma : 0;
        m_frame.m_predict          = m_predicting;
        ++m_frameVersion;
        m_assistedValid.store( assistedValid );
        m_candidateValid.store( !fault && !m_unlockedTraining.load() &&
                                ( !m_predicting || ready || probing || ( m_learning && m_gain != 0 ) ) );
        m_wfsReceipt.store( now );
        m_inputValid.store( true );
    }
    if( probing && !--m_exploration.front().m_count )
        m_exploration.erase( m_exploration.begin() );
    // One shared QR-RLS update per genuine WFS target, then every phase's coefficients and slot gains.
    m_multirate->learn();
    if( m_multirate->refresh( m_gainSets[m_gainBack] ) )
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        std::swap( m_gainBack, m_gainMiddle );
        m_gainFresh = true;
    }
    const auto done = synco::monotonicNs();
    m_learnUs.add( ( done - now ) / 1000. );
    if( m_lastFrameNs >= 0 && done - now > now - m_lastFrameNs )
        ++m_learnOverruns;
    m_lastFrameNs        = now;
    m_pending            = {};
    m_pending.m_index    = counter;
    m_pending.m_residual = measurement;
    m_pending.m_wfsValid = m_pending.m_timingValid = true;
    m_havePending                                  = true;
    m_wfsCounter                                   = counter;
    return 0;
}

inline bool loPredCtrlAcc_operational::outputValid() const
{
    const auto now = synco::monotonicNs(), last = m_wfsReceipt.load();
    const bool assisted = m_mode != DDSPC::ShadowMode::Standard;

    const char *reason = !m_inputValid.load()                                          ? "wfs_invalid"
                         : !m_candidateValid.load()                                    ? "command_invalid"
                         : last < 0 || now < last || now - last > m_inputTimeout * 1e9 ? "wfs_stale"
                         : assisted && !m_assistedValid.load()                         ? "acc_window"
                         : assisted && !m_syncLocked.load()                            ? "upstream_unlocked"
                                                                                       : nullptr;

    if( !reason )
    {
        m_wasValid.store( true );
        return true;
    }

    // Keep the condition that ended a qualified period, not the consequences that follow it.
    if( m_wasValid.exchange( false ) )
    {
        m_invalidWindow.store( m_windowReason.load() );
        m_invalidReason.store( reason );
    }

    return false;
}

inline bool loPredCtrlAcc_operational::outputCandidate( const synco::Request &request,
                                                        const synco::Record  &schedule,
                                                        std::span<float>      pixels )
{
    if( m_mode.load() == DDSPC::ShadowMode::AccelAssisted ) // Mode changes only after confirmed removal.
        return assistedCandidate( request, schedule, pixels );
    std::lock_guard<std::mutex> lock( m_commandMutex );
    if( pixels.size() != std::size_t( m_calibration.m_shapes.rows() ) || !m_slotPlan.cols() )
        return false;
    m_steps                = schedule[synco::steps];
    const auto   ordinal   = request.cycle * m_steps + request.step;
    std::int64_t slot      = 0; // Unknown base: the decision's first slot, as with a held command.
    const bool   sameEpoch = request.epoch == m_appliedEpoch && request.generation == m_appliedGeneration;
    if( m_planBase >= 0 && sameEpoch && ordinal > m_planBase )
        slot = std::min<std::int64_t>( ordinal - m_planBase - 1, m_slotPlan.cols() - 1 );
    Eigen::Map<DDSPC::Matrix>( pixels.data(), pixels.size(), 1 ).noalias() =
        m_calibration.m_shapes * m_slotPlan.col( slot );
    return true;
}

inline void loPredCtrlAcc_operational::fastSample( const DDSPC::AdcSample &sample )
{
    const auto counter = sample.m_frame.m_counter;
    const auto segment = ( sample.m_generation << 32 ) ^ sample.m_segment;
    if( m_fastHaveSample )
    {
        const auto &newest  = m_fastSamples[m_fastNewest % m_fastSamples.size()];
        const auto  current = ( newest.m_generation << 32 ) ^ newest.m_segment;
        if( counter <= m_fastNewest && segment == current )
            return; // Already held.
        if( counter != m_fastNewest + 1 || segment != current )
            m_fastHaveSample = false; // Discontinuity: only the new sample is valid.
    }
    if( !m_fastHaveSample )
        m_fastOldest = counter;
    m_fastSamples[counter % m_fastSamples.size()] = sample;
    m_fastNewest                                  = counter;
    m_fastHaveSample                              = true;
}

inline const DDSPC::AdcSample *loPredCtrlAcc_operational::fastSamples( std::int64_t now )
{
    if( m_adc )
    {
        std::array<DDSPC::AdcSample, 32> batch;
        for( int reads = 0; reads < 8; ++reads )
        {
            const auto copied = m_adc->copy_after( m_fastCursor, batch, now );
            m_fastCursor      = copied.m_next;
            if( copied.m_discontinuity || !copied.m_status.m_valid )
                m_fastHaveSample = false;
            for( std::size_t n = 0; n < copied.m_count && n < batch.size(); ++n )
                fastSample( batch[n] );
            if( !copied.m_more )
                break;
        }
    }
    return m_fastHaveSample ? &m_fastSamples[m_fastNewest % m_fastSamples.size()] : nullptr;
}

inline bool loPredCtrlAcc_operational::assistedCandidate( const synco::Request &request,
                                                          const synco::Record  &schedule,
                                                          std::span<float>      pixels )
{
    if( pixels.size() != std::size_t( m_calibration.m_shapes.rows() ) )
        return false;
    // Calculate as late as the render lead allows, so the newest ACC sample is used.
    // Sleep to 20 us before the render time, then poll the clock: sleep wake-up jitter stays out of the margin.
    const std::int64_t renderAt = request.cutoffNs - m_renderLeadNs;
    if( const auto wait = renderAt - synco::monotonicNs(); wait > 20000 )
        std::this_thread::sleep_for( std::chrono::nanoseconds( wait - 20000 ) );
    while( synco::monotonicNs() < renderAt )
    {
    }
    const auto start = synco::monotonicNs();
    const int  modes = m_settings.m_modes;
    int        prefix;
    bool       mismatch;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        m_steps = schedule[synco::steps];
        if( m_gainFresh )
        {
            std::swap( m_gainFront, m_gainMiddle );
            m_gainFresh = false;
        }
        if( m_fastVersion != m_frameVersion )
        {
            m_fastFrame.m_origin.m_state   = m_frame.m_origin.m_state; // Equal preallocated sizes: no allocation.
            m_fastFrame.m_origin.m_index   = m_frame.m_origin.m_index;
            m_fastFrame.m_origin.m_anchor  = m_frame.m_origin.m_anchor;
            m_fastFrame.m_origin.m_segment = m_frame.m_origin.m_segment;
            m_fastFrame.m_origin.m_valid   = m_frame.m_origin.m_valid;
            m_fastFrame.m_base             = m_frame.m_base;
            m_fastFrame.m_feedback         = m_frame.m_feedback;
            m_fastFrame.m_sigma            = m_frame.m_sigma;
            m_fastFrame.m_predict          = m_frame.m_predict;
            m_fastVersion                  = m_frameVersion;
        }
        if( m_fastClear )
        {
            m_fastLast.setZero();
            m_fastClear = false;
        }
        mismatch             = m_steps != m_slots;
        const auto ordinal   = request.cycle * m_steps + request.step;
        const bool sameEpoch = request.epoch == m_appliedEpoch && request.generation == m_appliedGeneration;
        prefix               = m_fastFrame.m_base >= 0 && sameEpoch
                                   ? int( std::clamp<std::int64_t>( ordinal - m_fastFrame.m_base - 1, -1, m_slots ) )
                                   : -1;
        // Executed slots of this frame, newest first: the scheduler's copies, or the retained copy for a slot that
        // was not copied (synco keeps the last copied contribution).
        for( int j = 0; j < prefix && prefix < m_slots; ++j )
        {
            const auto slot = std::min( m_fastFrame.m_base + 1 + j, m_appliedOrdinal );
            for( int mode = 0; mode < modes; ++mode )
                m_fastPrefix[( prefix - 1 - j ) * modes + mode] =
                    m_appliedOrdinal < 0 ? m_copiedCommand( mode ) : m_appliedSlots( mode, slot % m_ring );
        }
    }
    if( mismatch ) // The frame lift assumes dm_slots scheduler steps per WFS frame.
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        ++m_fastStats.m_mismatch;
        return false;
    }
    // Newest ACC samples since the frame anchor: phase k = new samples, at most r-1.
    const auto *newest = fastSamples( start );
    const auto &origin = m_fastFrame.m_origin;
    const auto  anchor = origin.m_anchor;
    int         phase = 0, available = 0;
    bool        anomaly = false;
    if( newest && origin.m_valid )
    {
        const auto segment = ( newest->m_generation << 32 ) ^ newest->m_segment;
        anomaly =
            segment != origin.m_segment || m_fastNewest < anchor || m_fastNewest - anchor >= std::uint64_t( m_ratio );
        if( segment == origin.m_segment && m_fastNewest > anchor )
            available = int( std::min<std::uint64_t>( m_fastNewest - anchor, m_ratio - 1 ) );
        // Every extra sample must belong to the current contiguous run and still be held.
        if( available && ( anchor + 1 < m_fastOldest || m_fastNewest - ( anchor + 1 ) >= m_fastSamples.size() ) )
            available = 0;
        for( ; phase < available; ++phase )
        {
            const auto &sample = m_fastSamples[( anchor + 1 + phase ) % m_fastSamples.size()];
            if( sample.m_frame.m_counter != anchor + 1 + phase )
                break;
            m_fastExtras[2 * phase]     = sample.m_acceleration[0];
            m_fastExtras[2 * phase + 1] = sample.m_acceleration[1];
        }
    }
    const auto &gains   = m_gainSets[m_gainFront];
    const bool  current = origin.m_valid && prefix >= 0 && prefix < m_slots;
    const bool  predict = current && m_fastFrame.m_predict && gains.m_ready;
    if( !m_fastFrame.m_predict )
        m_fastCommand.setZero();
    else if( !current )
        m_fastCommand = m_fastLast; // No frame owns this slot yet (late WFS): hold the last command.
    else
    {
        m_fastCommand = m_fastFrame.m_feedback;
        if( predict )
        {
            double command[16];
            DDSPC::MultirateDDSPC::command( m_multirate->config(),
                                            gains,
                                            origin,
                                            phase,
                                            prefix,
                                            m_fastExtras.data(),
                                            m_fastPrefix.data(),
                                            command );
            m_fastCommand += Eigen::Map<const DDSPC::VectorD>( command, modes );
        }
        if( m_fastFrame.m_sigma > 0 ) // Independent probes on every slot identify each slot's response.
            for( int mode = 0; mode < modes; ++mode )
                m_fastCommand( mode ) += m_fastFrame.m_sigma * m_fastDistribution( m_fastGenerator );
    }
    m_fastFloat = m_fastCommand.cast<float>();
    if( !m_fastFloat.allFinite() ) // Also rejects values beyond single precision.
    {
        m_fastCommand.setZero();
        m_fastFloat.setZero();
    }
    m_fastLast                                                             = m_fastCommand;
    Eigen::Map<DDSPC::Matrix>( pixels.data(), pixels.size(), 1 ).noalias() = m_calibration.m_shapes * m_fastFloat;
    const auto end                                                         = synco::monotonicNs();
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_commandMutex );
        auto                       &stats = m_fastStats;
        ++stats.m_slots;
        stats.m_holds += m_fastFrame.m_predict && !current;
        stats.m_anomalies += anomaly;
        if( predict )
        {
            ++stats.m_predictions;
            ++stats.m_phase[phase];
            stats.m_lagging += phase < prefix;
            stats.m_compute.add( ( end - start ) / 1000. );
            if( request.cutoffNs > 0 )
            {
                stats.m_margin.add( ( request.cutoffNs - end ) / 1000. );
                stats.m_wake.add( std::max<std::int64_t>( 0, start - renderAt ) / 1000. );
            }
            if( newest && newest->m_acquiredNs > 0 )
            {
                stats.m_ageCalc.add( ( start - newest->m_acquiredNs ) / 1000. );
                if( request.targetNs > 0 )
                    stats.m_ageApply.add( ( request.targetNs - newest->m_acquiredNs ) / 1000. );
            }
        }
    }
    return true;
}

inline void loPredCtrlAcc_operational::copied( const synco::CommitRecord &commit, std::span<const float> pixels )
{
    std::lock_guard<std::mutex>     lock( m_commandMutex );
    Eigen::Map<const DDSPC::Matrix> image( pixels.data(), pixels.size(), 1 );
    m_copiedCommand.noalias() = m_calibration.m_projection * image;
    m_commandKnown            = m_copiedCommand.allFinite();
    const auto ordinal        = commit[synco::cmdCycle] * m_steps + commit[synco::cmdStep];
    if( commit[synco::cmdEpoch] != m_appliedEpoch || commit[synco::cmdGeneration] != m_appliedGeneration ||
        m_appliedOrdinal < 0 || ordinal <= m_appliedOrdinal || ordinal - m_appliedOrdinal > m_ring )
    {
        // New scheduler incarnation or a long gap: restart slot history at this copy.
        m_appliedEpoch      = commit[synco::cmdEpoch];
        m_appliedGeneration = commit[synco::cmdGeneration];
        m_planBase = m_decisionOrdinal = -1;
        m_slotRestart                  = true;
    }
    else // Missed or discarded slots retain the previous copy at the destination.
        for( auto skipped = m_appliedOrdinal + 1; skipped < ordinal; ++skipped )
            m_appliedSlots.col( skipped % m_ring ) = m_appliedSlots.col( m_appliedOrdinal % m_ring );
    m_appliedSlots.col( ordinal % m_ring ) = m_copiedCommand;
    m_appliedOrdinal                       = ordinal;
}

inline int loPredCtrlAcc_operational::allocate( const dev::shmimT & )
{
    m_inputValid.store( false );
    if( shmimMonitorT::m_dataType != _DATATYPE_FLOAT ||
        ( shmimMonitorT::m_width != 1 && shmimMonitorT::m_height != 1 ) ||
        shmimMonitorT::m_width * shmimMonitorT::m_height <=
            *std::max_element( m_calibration.m_indices.begin(), m_calibration.m_indices.begin() + m_settings.m_modes ) )
        return -1;
    { //mutex scope
        const std::size_t size = std::size_t( shmimMonitorT::m_width ) * shmimMonitorT::m_height;
        m_modalScratch.assign( size, 0 );
        m_modalCounter = 0;
        std::lock_guard<std::mutex> lock( m_modalMutex );
        if( m_modalFrame.size() != size )
        {
            // A changed input shape makes the publisher reattach or recreate its stream at the new size.
            frameGrabberT::m_reconfig = !m_modalFrame.empty();
            m_modalFrame.assign( size, 0 );
            m_modalFresh = false;
        }
    }
    std::lock_guard<std::mutex> lock( m_modelMutex );
    m_models->invalidate( {} );
    m_havePending = false;
    m_wfsCounter  = 0;
    return 0;
}

inline int loPredCtrlAcc_operational::processImage( void *source, const dev::shmimT & )
{
    DDSPC::Matrix applied;
    std::int64_t  now, ordinal;
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        now     = synco::monotonicNs();
        ordinal = appliedSlots( applied );
    }
    const auto &metadata = *shmimMonitorT::m_imageStream.md;
    const auto  counter  = metadata.cnt0;
    if( metadata.write || !source )
        return 0;
    if( metadata.naxis == 3 )
    {
        const auto plane = metadata.cnt1;
        if( plane >= metadata.size[2] ||
            source != shmimMonitorT::m_imageStream.array.F +
                          plane * std::size_t( shmimMonitorT::m_width ) * shmimMonitorT::m_height )
            return 0;
    }
    std::atomic_thread_fence( std::memory_order_acquire );
    DDSPC::Matrix measurement( m_settings.m_modes, 1 ), features;
    for( int mode = 0; mode < m_settings.m_modes; ++mode )
        measurement( mode ) = static_cast<const float *>( source )[m_calibration.m_indices[mode]];
    // The modal output copies the same coherent frame; the input and the measurement above are never modified.
    const bool     modal = !m_modalScratch.empty(); // Sized by allocate().
    const timespec atime = metadata.atime;
    if( modal )
        std::memcpy( m_modalScratch.data(), source, m_modalScratch.size() * sizeof( float ) );
    std::atomic_thread_fence( std::memory_order_acquire );
    if( metadata.write || counter != metadata.cnt0 )
        return 0;
    if( modal ) // Before learning, as loPredCtrl posts its output before updating the model.
        publishModal( counter, atime );
    const bool valid = acceleration( features, now );
    return step( measurement, counter, features, valid, applied, ordinal, now );
}

inline void loPredCtrlAcc_operational::validateModalOutput() const
{
    const auto &name = frameGrabberT::m_shmimName;
    if( m_settings.m_modes != 2 || m_calibration.m_indices.size() < 2 || m_calibration.m_indices[0] != 0 ||
        m_calibration.m_indices[1] != 1 )
        throw std::invalid_argument( "The main-loop modal output requires DDSPC to control exactly modes 0 and 1" );
    if( name.empty() || name == shmimMonitorT::m_shmimName || name == m_syncopationTiming || name.ends_with( "_aux" ) )
        throw std::invalid_argument(
            "framegrabber.shmimName must name a modal stream distinct from the WFS input and DM transport" );
}

inline bool loPredCtrlAcc_operational::modalZeroed( const SyncoOutputStatus &output, bool engaged )
{
    return output.m_requested || ( engaged && !output.m_zeroConfirmed );
}

inline void loPredCtrlAcc_operational::publishModal( std::uint64_t counter, const timespec &atime )
{
    if( counter == m_modalCounter )
    {
        return;
    }

    if( modalZeroed( m_output.status(), m_modalEngaged.load() ) )
    {
        for( int mode = 0; mode < m_settings.m_modes; ++mode ) // Validated to be exactly modes 0 and 1.
        {
            m_modalScratch[m_calibration.m_indices[mode]] = 0;
        }
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_modalMutex );
        if( m_modalFrame.size() != m_modalScratch.size() )
            return;
        m_modalFrame.swap( m_modalScratch ); // Equal preallocated sizes: no allocation.
        m_modalAtime = atime;
        m_modalFresh = true;
    }
    m_modalCounter = counter;
    sem_post( &m_modalSemaphore );
}

inline int loPredCtrlAcc_operational::modalRefused( const std::string &reason )
{
    if( !m_modalLogged )
        log<text_log>( "Modal output " + frameGrabberT::m_shmimName + " not published: " + reason,
                       logPrio::LOG_WARNING );
    m_modalLogged = true;
    return -1;
}

inline int loPredCtrlAcc_operational::configureAcquisition()
{
    std::size_t size;
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_modalMutex );
        size = m_modalFrame.size();
    }
    if( !size )
        return -1; // Wait for the WFS input shape.
    char path[4096];
    if( ImageStreamIO_filename( path, sizeof( path ), frameGrabberT::m_shmimName.c_str() ) != 0 )
        return modalRefused( "invalid stream name" );
    if( m_modalLock < 0 )
    {
        // The syncopation producers' cooperative convention: one publisher per stream.
        const std::string lockName = std::string( path ) + ".producer";
        const int         lock     = open( lockName.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0660 );
        if( lock < 0 || flock( lock, LOCK_EX | LOCK_NB ) != 0 )
        {
            if( lock >= 0 )
                close( lock );
            return modalRefused( "another publisher holds " + lockName );
        }
        m_modalLock = lock;
    }
    struct stat info{};
    if( stat( path, &info ) != 0 )
    {
        // No stream yet: the frameGrabber creates it at this size.
        frameGrabberT::m_width    = size;
        frameGrabberT::m_height   = 1;
        frameGrabberT::m_dataType = _DATATYPE_FLOAT;
        m_modalLogged             = false;
        return 0;
    }
    // Attach an existing stream, so its consumers keep their mapping. Never recreate, resize or share it.
    if( frameGrabberT::openShmim() != 0 )
        return modalRefused( "existing stream is not ready" );
    const auto *metadata = frameGrabberT::m_imageStream->md;
    const auto  before   = metadata->cnt0;
    std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) ); // 40 frames at 2 kHz.
    const char *refusal =
        frameGrabberT::m_width != size || frameGrabberT::m_height != 1 || metadata->datatype != _DATATYPE_FLOAT
            ? "existing stream has a different shape or type"
        : metadata->cnt0 != before ? "existing stream is being published by another process"
                                   : nullptr;
    if( refusal )
    {
        ImageStreamIO_closeIm( frameGrabberT::m_imageStream );
        free( frameGrabberT::m_imageStream );
        frameGrabberT::m_imageStream = nullptr;
        return modalRefused( refusal );
    }
    m_modalLogged = false;
    return 0;
}

inline float loPredCtrlAcc_operational::fps()
{
    return 0;
}

inline int loPredCtrlAcc_operational::startAcquisition()
{
    return 0;
}

inline int loPredCtrlAcc_operational::acquireAndCheckValid()
{
    timespec ts;
    if( clock_gettime( CLOCK_REALTIME, &ts ) < 0 )
        return log<software_critical, -1>( { errno, "clock_gettime" } );
    ts.tv_sec += 1;
    if( sem_timedwait( &m_modalSemaphore, &ts ) != 0 )
        return 1;
    std::lock_guard<std::mutex> lock( m_modalMutex );
    if( m_modalFrame.size() != std::size_t( frameGrabberT::m_width ) * frameGrabberT::m_height )
    {
        frameGrabberT::m_reconfig = true; // Input shape changed: reattach before publishing.
        return 1;
    }
    return m_modalFresh ? 0 : 1;
}

inline int loPredCtrlAcc_operational::loadImageIntoStream( void *dest )
{
    std::lock_guard<std::mutex> lock( m_modalMutex );
    const auto count = std::min( m_modalFrame.size(), std::size_t( frameGrabberT::m_width ) * frameGrabberT::m_height );
    std::memcpy( dest, m_modalFrame.data(), count * sizeof( float ) );
    frameGrabberT::m_currImageTimestamp = m_modalAtime; // Stored as the output atime after this call.
    m_modalFresh                        = false;
    return 0;
}

inline int loPredCtrlAcc_operational::reconfig()
{
    return 0;
}

inline int loPredCtrlAcc_operational::checkRecordTimes()
{
    return telemeterT::checkRecordTimes( telem_fgtimings() );
}

inline int loPredCtrlAcc_operational::recordTelem( const telem_fgtimings * )
{
    return recordFGTimings( true );
}

inline bool loPredCtrlAcc_operational::removed() const
{
    const auto output = m_output.status();
    return !output.m_requested && output.m_zeroConfirmed;
}

inline void loPredCtrlAcc_operational::stopPrediction()
{
    m_output.request( false ); // Off is always accepted and schedules removal of the DDSPC contribution.
    m_predicting      = false;
    m_predictRejected = false;
    updateSwitchIfChanged( m_indiP_predicting, "toggle", pcf::IndiElement::Off, INDI_OK );
}

inline std::vector<loPredCtrlAcc_operational::Stage>
loPredCtrlAcc_operational::parseExploration( const std::string &text )
{
    std::vector<Stage> stages;
    std::stringstream  input( text );
    std::string        count, sigma, lambda;
    while( std::getline( input, count, ',' ) )
    {
        if( !std::getline( input, sigma, ',' ) || !std::getline( input, lambda, ',' ) || stages.size() >= 32 )
            throw std::invalid_argument( "Expected count,sigma,regularization exploration triples" );
        std::size_t nc, ns, nl;
        const auto  n = std::stoul( count, &nc );
        const auto  s = std::stof( sigma, &ns ), l = std::stof( lambda, &nl );
        if( nc != count.size() || ns != sigma.size() || nl != lambda.size() || n == 0 || n > 1000000 ||
            !std::isfinite( s ) || s < 0 || !std::isfinite( l ) || l <= 0 )
            throw std::invalid_argument( "Invalid exploration count, strength or regularization" );
        stages.push_back( { unsigned( n ), s, l } );
    }
    if( stages.empty() || text.back() == ',' )
        throw std::invalid_argument( "Empty or incomplete exploration sequence" );
    return stages;
}

inline int loPredCtrlAcc_operational::setupINDI()
{
    createStandardIndiToggleSw( m_indiP_learning, "learn", "Learning", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_learning, INDI_NEWCALLBACK( m_indiP_learning ) ) < 0 )
        return -1;
    createStandardIndiToggleSw( m_indiP_predicting, "predict", "Prediction", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_predicting, INDI_NEWCALLBACK( m_indiP_predicting ) ) < 0 )
        return -1;
    createStandardIndiToggleSw( m_indiP_learnStd, "learn_std", "Standard learning sequence", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_learnStd, INDI_NEWCALLBACK( m_indiP_learnStd ) ) < 0 )
        return -1;
    createStandardIndiRequestSw( m_indiP_reset, "reset_model", "Reset models", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_reset, INDI_NEWCALLBACK( m_indiP_reset ) ) < 0 )
        return -1;
    createStandardIndiRequestSw( m_indiP_zeroDm, "zero_dm", "Zero DDSPC DM channel", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_zeroDm, INDI_NEWCALLBACK( m_indiP_zeroDm ) ) < 0 )
    {
        return -1;
    }
    createStandardIndiRequestSw( m_indiP_releaseTT, "release_tt", "Return tip/tilt to main loop", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_releaseTT, INDI_NEWCALLBACK( m_indiP_releaseTT ) ) < 0 )
    {
        return -1;
    }
    createStandardIndiSelectionSw( m_indiP_mode, "ddspc_mode", { "standard", "accel_assisted" }, "Mode", "DDSPC" );
    m_indiP_mode["standard"].setSwitchState( pcf::IndiElement::On );
    if( registerIndiPropertyNew( m_indiP_mode, INDI_NEWCALLBACK( m_indiP_mode ) ) < 0 )
        return -1;
    createStandardIndiNumber<float>( m_indiP_gain, "gain", -10, 10, .001, "%0.4f" );
    m_indiP_gain["current"].set( m_gain );
    m_indiP_gain["target"].set( m_gain );
    if( registerIndiPropertyNew( m_indiP_gain, INDI_NEWCALLBACK( m_indiP_gain ) ) < 0 )
        return -1;
    createStandardIndiText( m_indiP_exploration, "exploration_sequence", "Exploration sequence", "DDSPC" );
    if( registerIndiPropertyNew( m_indiP_exploration, INDI_NEWCALLBACK( m_indiP_exploration ) ) < 0 )
        return -1;
    createROIndiText( m_indiP_control, "ddspc_control", "requested_mode" );
    for( const auto &key : { "standard_ready",
                             "assisted_ready",
                             "learning",
                             "prediction",
                             "accel_sync",
                             "standard_state",
                             "assisted_state",
                             "main_loop_tt" } )
    {
        m_indiP_control.add( pcf::IndiElement( key, "false" ) );
    }
    if( registerIndiPropertyReadOnly( m_indiP_control ) < 0 )
        return -1;
    createROIndiNumber( m_indiP_performance, "ddspc_performance" );
    for( const auto &key :
         { "standard_updates",     "assisted_updates",    "standard_rmse",         "assisted_rmse",
           "transport_published",  "transport_acks",      "transport_skipped",     "slot_shifts",
           "publish_rate_hz",      "predict_rate_hz",     "update_rate_hz",        "refresh_rate_hz",
           "fast_predictions",     "fast_holds",          "acc_lagging",           "acc_anomalies",
           "step_mismatch",        "fast_us_mean",        "fast_us_p99",           "fast_us_max",
           "margin_us_min",        "margin_us_mean",      "wake_us_p99",           "acc_age_calc_us_mean",
           "acc_age_calc_us_p99",  "acc_age_calc_us_max", "acc_age_apply_us_mean", "acc_age_apply_us_p99",
           "acc_age_apply_us_max", "learn_us_mean",       "learn_us_p99",          "learn_us_max",
           "learn_overruns" } )
        m_indiP_performance.add( pcf::IndiElement( key, 0.0 ) );
    m_phaseNames.clear();
    for( int k = 0; k < m_ratio; ++k )
        for( const auto *key : { "rmse_phase", "acc_phase" } )
        {
            m_phaseNames.push_back( key + std::to_string( k ) );
            m_indiP_performance.add( pcf::IndiElement( m_phaseNames.back(), 0.0 ) );
        }
    if( registerIndiPropertyReadOnly( m_indiP_performance ) < 0 )
        return -1;
    createROIndiText( m_indiP_safety, "safety", "output_state" );
    for( const auto &key : { "dm_write_enabled", "zero_confirmed" } )
        m_indiP_safety.add( pcf::IndiElement( key, "false" ) );
    if( registerIndiPropertyReadOnly( m_indiP_safety ) < 0 )
        return -1;
    createROIndiText( m_indiP_qualification, "assisted_qualification", "blockers" );
    for( const auto &key : { "upstream_enabled",
                             "upstream_locked",
                             "upstream_phase_error_us",
                             "upstream_read_latency_error_us",
                             "upstream_age_s",
                             "adc_state",
                             "acc_window",
                             "model",
                             "diagnostic_mode" } )
        m_indiP_qualification.add( pcf::IndiElement( key, "" ) );
    if( registerIndiPropertyReadOnly( m_indiP_qualification ) < 0 )
        return -1;
    createROIndiNumber( m_indiP_adc, "adc_values" );
    for( const auto &key : { "raw0", "raw1", "accel0", "accel1", "rate_hz", "valid" } )
        m_indiP_adc.add( pcf::IndiElement( key, 0.0 ) );
    if( registerIndiPropertyReadOnly( m_indiP_adc ) < 0 )
        return -1;
    return 0;
}

inline int loPredCtrlAcc_operational::appStartup()
{
    if( m_shutdown || !m_models || !m_calibration.m_shapes.rows() || setupINDI() < 0 )
        return -1;
    try
    {
        if( m_accelEnabled )
        {
            try
            {
                if( m_syncDevice.empty() || ( m_syncProperty != "timingDiag" && m_syncProperty != "ddspcTiming" ) )
                    throw std::invalid_argument( "Configure the existing upstream ADC lock diagnostic" );
                m_adc = std::make_unique<DDSPC::AdcReader>( m_adcConfig );
                m_adc->start();
                if( registerIndiPropertySet(
                        m_indiP_sync, m_syncDevice, m_syncProperty, INDI_SETCALLBACK( m_indiP_sync ) ) < 0 )
                    throw std::runtime_error( "Could not subscribe to upstream synchronization" );
            }
            catch( const std::exception &error )
            {
                m_adc.reset();
                log<text_log>( std::string( "ACC assistance unavailable; standard DDSPC remains available: " ) +
                               error.what() );
            }
        }
        SyncoOutputConfig output;
        // Operator predict.toggle and the existing producer protocol supply intent/ownership. No extra trust switches.
        output.m_authorized = output.m_routingVerified = output.m_producerDefaults = true;
        output.m_width                                                             = m_calWidth;
        output.m_height                                                            = m_calHeight;
        m_output.configure( output, m_channel, m_syncopationTiming, m_writeLeadUs, m_calibration.m_shapes.rows() );
        m_output.start(
            [this] { return outputValid(); },
            [this](
                const synco::Request &request, const synco::Record &schedule, std::int64_t, std::span<float> pixels )
            { return outputCandidate( request, schedule, pixels ); },
            [this]( const synco::CommitRecord &commit, std::span<const float> pixels ) { copied( commit, pixels ); },
            false );
        // The dmSpeckle worker pattern: framework thread with configured real-time priority and cpuset.
        if( threadStart( m_outputThread,
                         m_outputThreadInit,
                         m_outputThreadID,
                         m_outputThreadProp,
                         m_outputThreadPrio,
                         m_outputThreadCpuset,
                         "dm",
                         this,
                         outputThreadStart ) < 0 )
            throw std::runtime_error( "DM worker thread priority/cpuset setup failed" );
        m_output.initialize();
        // Before WFS monitoring, which posts the semaphore.
        if( sem_init( &m_modalSemaphore, 0, 0 ) < 0 )
            throw std::runtime_error( "Could not initialize the modal-output semaphore" );
        FRAMEGRABBER_APP_STARTUP;
        TELEMETER_APP_STARTUP;
        SHMIMMONITOR_APP_STARTUP;
    }
    catch( const std::exception &error )
    {
        return log<software_error, -1>( { __FILE__, __LINE__, error.what() } );
    }
    state( stateCodes::OPERATING );
    return 0;
}

inline int loPredCtrlAcc_operational::appLogic()
{
    SHMIMMONITOR_APP_LOGIC;
    FRAMEGRABBER_APP_LOGIC;
    TELEMETER_APP_LOGIC;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    SHMIMMONITOR_UPDATE_INDI;
    FRAMEGRABBER_UPDATE_INDI;
    const auto             output = m_output.status();
    DDSPC::ShadowStatus    standard;
    DDSPC::MultirateStatus assisted;
    bool                   learning, predicting;
    std::uint64_t          shifts, overruns, needed;
    Latency                learn;
    FastStats              fast;
    { //mutex scope
        std::lock_guard<std::mutex> modelLock( m_modelMutex );
        if( !output.m_requested )
            m_predicting = false;

        if( !output.m_requested && output.m_zeroConfirmed )
        {
            m_modalEngaged.store( false ); // DDSPC's contribution is removed: main-loop tip/tilt passes through.
        }

        standard   = m_models->status( DDSPC::ShadowMode::Standard );
        assisted   = m_multirate->status();
        learning   = m_learning;
        predicting = m_predicting;
        learn      = m_learnUs;
        overruns   = m_learnOverruns;
        needed     = m_multirate->config().features( m_ratio - 1 );
        m_learnUs  = {};
    }
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        shifts = m_slotShifts;
        fast   = m_fastStats;
        for( auto *latency : { &m_fastStats.m_compute,
                               &m_fastStats.m_margin,
                               &m_fastStats.m_wake,
                               &m_fastStats.m_ageCalc,
                               &m_fastStats.m_ageApply } )
            *latency = {};
    }
    const auto       now   = synco::monotonicNs();
    const auto       last  = m_wfsReceipt.load();
    const bool       fresh = m_inputValid.load() && last >= 0 && now >= last && now - last <= m_inputTimeout * 1e9;
    const bool       valid = outputValid();
    DDSPC::AdcStatus adc;
    if( m_adc )
    {
        adc                = m_adc->status();
        const auto &sample = adc.m_latest;
        updatesIfChanged<double>( m_indiP_adc,
                                  { "raw0", "raw1", "accel0", "accel1", "rate_hz", "valid" },
                                  { double( sample.m_frame.m_raw[0] ),
                                    double( sample.m_frame.m_raw[1] ),
                                    sample.m_acceleration[0],
                                    sample.m_acceleration[1],
                                    adc.m_rateHz,
                                    double( adc.m_valid ) },
                                  adc.m_valid ? INDI_OK : INDI_IDLE );
    }
    const bool selected      = m_mode.load() == DDSPC::ShadowMode::Standard;
    const bool selectedReady = selected ? standard.m_ready : assisted.m_ready;
    const bool locked = m_syncLocked.load(), samples = m_assistedValid.load();
    // Every failing assisted-mode condition, in dependence order; the first is reported as assisted_state.
    std::string blockers;
    auto        block      = [&]( const std::string &reason ) { blockers += ( blockers.empty() ? "" : "," ) + reason; };
    const auto  upstreamNs = m_upstreamNs.load();
    const bool  adcProperty = m_syncProperty == "ddspcTiming";
    const auto  source      = m_syncDevice + "." + m_syncProperty + ".";
    const int   enabled = m_upstreamEnabled.load(), upstreamLock = m_upstreamLocked.load();
    if( !fresh )
        block( "wfs_stale" );
    if( !m_accelEnabled || !m_adc )
        block( "adc_disabled" );
    else if( !adc.m_valid )
        block( std::string( "adc_" ) + DDSPC::adc_state( adc.m_state ) );
    if( upstreamNs < 0 )
        block( "no_upstream_status_from_" + m_syncDevice + "." + m_syncProperty );
    else
    {
        if( enabled != 1 )
            block( source + ( adcProperty ? "enabled=" : "mode_code=" ) + std::to_string( enabled ) );
        if( upstreamLock != 1 )
            block( source + ( adcProperty ? "valid=" : "delay_lock=" ) + std::to_string( upstreamLock ) );
    }
    if( !samples )
        block( std::string( "acc_window_" ) + m_windowReason.load() );
    if( m_unlockedTraining.load() )
        block( "reset_required_trained_without_upstream_lock" );
    if( !assisted.m_ready )
        block( std::string( assisted.m_training ? "model_warming_" : "model_untrained_" ) +
               std::to_string( assisted.m_updates ) + "_of_" + std::to_string( needed ) + "_updates" );
    if( fast.m_mismatch > m_rates.m_mismatch )
        block( "scheduler_steps_differ_from_dm_slots" );
    m_rates.m_mismatch = fast.m_mismatch;
    auto format        = []( double value )
    {
        char text[32];
        std::snprintf( text, sizeof( text ), "%.1f", value );
        return std::isfinite( value ) ? std::string( text ) : std::string( "n/a" );
    };
    const std::string assistedState = blockers.empty() ? "ready" : blockers.substr( 0, blockers.find( ',' ) );
    updatesIfChanged<std::string>( m_indiP_control,
                                   { "requested_mode",
                                     "standard_ready",
                                     "assisted_ready",
                                     "learning",
                                     "prediction",
                                     "accel_sync",
                                     "standard_state",
                                     "assisted_state",
                                     "main_loop_tt" },
                                   { selected ? "standard" : "accel_assisted",
                                     standard.m_ready && fresh ? "true" : "false",
                                     assisted.m_ready && fresh && samples && locked ? "true" : "false",
                                     learning ? "true" : "false",
                                     !predicting             ? "off"
                                     : !valid                ? "inhibited"
                                     : !selectedReady        ? "warming"
                                     : output.m_acknowledged ? "active"
                                                             : "awaiting_copy",
                                     locked ? "true" : "false",
                                     !fresh             ? "awaiting_wfs"
                                     : standard.m_ready ? "ready"
                                                        : DDSPC::shadow_reason( standard.m_reason ),
                                     assistedState,
                                     modalZeroed( output, m_modalEngaged.load() ) ? "zeroed" : "passthrough" },
                                   INDI_OK );
    updatesIfChanged<std::string>( m_indiP_qualification,
                                   { "blockers",
                                     "upstream_enabled",
                                     "upstream_locked",
                                     "upstream_phase_error_us",
                                     "upstream_read_latency_error_us",
                                     "upstream_age_s",
                                     "adc_state",
                                     "acc_window",
                                     "model",
                                     "diagnostic_mode" },
                                   { blockers.empty() ? "none" : blockers,
                                     std::to_string( enabled ),
                                     std::to_string( upstreamLock ),
                                     format( m_upstreamPhaseUs.load() ),
                                     format( m_upstreamLatencyUs.load() ),
                                     upstreamNs < 0 ? "n/a" : format( ( now - upstreamNs ) / 1e9 ),
                                     !m_adc ? "disabled" : DDSPC::adc_state( adc.m_state ),
                                     m_windowReason.load(),
                                     std::to_string( assisted.m_updates ) + "/" + std::to_string( needed ) +
                                         ( assisted.m_ready      ? " ready"
                                           : assisted.m_training ? " learning"
                                                                 : " frozen" ),
                                     !m_assistedDiagnostic       ? "off"
                                     : m_unlockedTraining.load() ? "on_learned_without_lock"
                                                                 : "on" },
                                   blockers.empty() ? INDI_OK : INDI_BUSY );
    // Rates over the interval since the previous update.
    const double seconds = m_rates.m_ns > 0 ? ( now - m_rates.m_ns ) / 1e9 : 0;
    auto         rate    = [seconds]( std::uint64_t current, std::uint64_t &previous )
    {
        const double value = seconds > 0 ? ( current - previous ) / seconds : 0;
        previous           = current;
        return value;
    };
    const double publishRate = rate( output.m_published, m_rates.m_published );
    const double predictRate = rate( fast.m_predictions, m_rates.m_predictions );
    const double updateRate  = rate( assisted.m_updates, m_rates.m_updates );
    const double refreshRate = rate( assisted.m_refreshes, m_rates.m_refreshes );
    m_rates.m_ns             = now;
    std::vector<const char *> names{
        "standard_updates",     "assisted_updates",    "standard_rmse",         "assisted_rmse",
        "transport_published",  "transport_acks",      "transport_skipped",     "slot_shifts",
        "publish_rate_hz",      "predict_rate_hz",     "update_rate_hz",        "refresh_rate_hz",
        "fast_predictions",     "fast_holds",          "acc_lagging",           "acc_anomalies",
        "step_mismatch",        "fast_us_mean",        "fast_us_p99",           "fast_us_max",
        "margin_us_min",        "margin_us_mean",      "wake_us_p99",           "acc_age_calc_us_mean",
        "acc_age_calc_us_p99",  "acc_age_calc_us_max", "acc_age_apply_us_mean", "acc_age_apply_us_p99",
        "acc_age_apply_us_max", "learn_us_mean",       "learn_us_p99",          "learn_us_max",
        "learn_overruns" };
    std::vector<double> values{ double( standard.m_updates ),
                                double( assisted.m_updates ),
                                standard.m_originRmse,
                                assisted.m_rmse.empty() ? 0 : assisted.m_rmse.back(),
                                double( output.m_published ),
                                double( output.m_acknowledgements ),
                                double( output.m_skipped ),
                                double( shifts ),
                                publishRate,
                                predictRate,
                                updateRate,
                                refreshRate,
                                double( fast.m_predictions ),
                                double( fast.m_holds ),
                                double( fast.m_lagging ),
                                double( fast.m_anomalies ),
                                double( fast.m_mismatch ),
                                fast.m_compute.mean(),
                                fast.m_compute.quantile( .99 ),
                                fast.m_compute.m_max,
                                fast.m_margin.m_min,
                                fast.m_margin.mean(),
                                fast.m_wake.quantile( .99 ),
                                fast.m_ageCalc.mean(),
                                fast.m_ageCalc.quantile( .99 ),
                                fast.m_ageCalc.m_max,
                                fast.m_ageApply.mean(),
                                fast.m_ageApply.quantile( .99 ),
                                fast.m_ageApply.m_max,
                                learn.mean(),
                                learn.quantile( .99 ),
                                learn.m_max,
                                double( overruns ) };
    for( int k = 0; k < m_ratio; ++k )
    {
        names.push_back( m_phaseNames[2 * k].c_str() );
        values.push_back( k < int( assisted.m_rmse.size() ) ? assisted.m_rmse[k] : 0 );
        names.push_back( m_phaseNames[2 * k + 1].c_str() );
        values.push_back( k < int( fast.m_phase.size() ) ? double( fast.m_phase[k] ) : 0 );
    }
    updatesIfChanged<double>( m_indiP_performance, names, values, INDI_OK );
    if( output.m_faults != m_faultsLogged )
    {
        m_faultsLogged = output.m_faults;
        m_faultDetail  = output.m_faultReason;

        // A lost qualification names the outputValid() condition, and for the ACC window its reason.
        if( std::string( output.m_faultReason ) == "input_or_model_qualification_lost" )
        {
            const std::string condition = m_invalidReason.load();
            m_faultDetail += ": " + condition;

            if( condition == "acc_window" )
            {
                m_faultDetail += std::string( " " ) + m_invalidWindow.load();
            }
        }

        log<text_log>( "DDSPC output fault, prediction stopped: " + m_faultDetail, logPrio::LOG_WARNING );
    }

    updatesIfChanged<std::string>( m_indiP_safety,
                                   { "output_state", "dm_write_enabled", "zero_confirmed" },
                                   { output.m_fault && m_faultDetail != output.m_reason
                                         ? std::string( output.m_reason ) + " (" + m_faultDetail + ")"
                                         : std::string( output.m_reason ),
                                     output.m_requested && !output.m_fault ? "true" : "false",
                                     output.m_zeroConfirmed ? "true" : "false" },
                                   output.m_fault ? INDI_ALERT : INDI_OK );
    updateSwitchIfChanged( m_indiP_predicting,
                           "toggle",
                           predicting ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           output.m_fault || m_predictRejected ? INDI_ALERT : INDI_OK );
    return 0;
}

inline int loPredCtrlAcc_operational::appShutdown()
{
    const bool removed = stopOutput();
    frameGrabberT::appShutdown();
    telemeterT::appShutdown();
    if( m_modalLock >= 0 )
    {
        close( m_modalLock );
        m_modalLock = -1;
    }
    SHMIMMONITOR_APP_SHUTDOWN;
    m_adc.reset();
    if( !removed )
        log<software_error>(
            { __FILE__, __LINE__, "Own-channel zero acknowledgement unavailable; removal unconfirmed" } );
    return removed ? 0 : -1;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_learning )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_learning, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    const auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_learning = value == pcf::IndiElement::On;
    applyTraining();
    if( !m_learning )
    {
        m_exploration.clear();
        updateSwitchIfChanged( m_indiP_learnStd, "toggle", pcf::IndiElement::Off, INDI_OK );
    }
    updateSwitchIfChanged( m_indiP_learning, "toggle", value, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_predicting )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_predicting, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    const auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    const bool                  enabled = value == pcf::IndiElement::On;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    if( enabled && !m_predicting )
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        m_slotPlan.setZero();
        m_fastClear = true;
    }
    // Refused if learned without upstream lock (reset the models first), or with nothing to apply yet.
    const bool unlocked = m_mode.load() == DDSPC::ShadowMode::AccelAssisted && m_unlockedTraining.load();
    const bool allowed =
        !enabled || ( !unlocked && ( selectedReady() || ( m_learning && ( m_gain != 0 || !m_exploration.empty() ) ) ) );
    const bool accepted = allowed && m_output.request( enabled );
    m_predictRejected   = !accepted;

    if( accepted )
    {
        m_predicting = enabled;
    }

    if( m_predicting )
    {
        m_modalEngaged.store( true ); // Only an accepted request engages; the request itself already zeroes.
    }

    // A refusal is always answered, so a client never waits for a readback that will not come.
    updateSwitchIfChanged( m_indiP_predicting,
                           "toggle",
                           m_predicting ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           accepted ? INDI_OK : INDI_ALERT );
    return accepted ? 0 : -1;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_learnStd )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_learnStd, ipRecv );
    if( !ipRecv.find( "toggle" ) )
        return -1;
    const auto value = ipRecv["toggle"].getSwitchState();
    if( value != pcf::IndiElement::On && value != pcf::IndiElement::Off )
        return -1;
    const bool                  enabled = value == pcf::IndiElement::On;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    if( enabled && !m_predicting )
    { //mutex scope
        std::lock_guard<std::mutex> commandLock( m_commandMutex );
        m_slotPlan.setZero();
        m_fastClear = true;
    }
    if( enabled && !m_output.request( true ) )
    {
        m_predictRejected = true;
        updateSwitchIfChanged( m_indiP_learnStd, "toggle", m_indiP_learnStd["toggle"].getSwitchState(), INDI_ALERT );
        return -1;
    }

    if( enabled )
    {
        m_predictRejected = false;
    }

    m_learning = enabled;
    if( enabled )
    {
        m_exploration = parseExploration( "1000,0.05,100,1000,0.05,3,1000,0.05,0.1" );
        m_predicting  = true;
        m_modalEngaged.store( true );
    }
    else
        m_exploration.clear();
    applyTraining();
    updateSwitchIfChanged( m_indiP_learning, "toggle", value, INDI_OK );
    updateSwitchIfChanged( m_indiP_learnStd, "toggle", value, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_reset )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_reset, ipRecv );
    if( !ipRecv.find( "request" ) || ipRecv["request"].getSwitchState() != pcf::IndiElement::On )
        return 0;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( !removed() || !m_output.clear_fault() )
        return -1;
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_predicting = false;
    m_exploration.clear();
    resetModels();
    updateSwitchIfChanged( m_indiP_reset, "request", pcf::IndiElement::Off, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_zeroDm )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_zeroDm, ipRecv );
    if( !ipRecv.find( "request" ) || ipRecv["request"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    stopPrediction();

    // A confirmed removal is written again, as at startup; this never withholds main-loop tip/tilt by itself.
    if( !m_output.initialize() )
    {
        return -1;
    }

    log<text_log>( "DDSPC DM channel " + m_channel + " zero requested", logPrio::LOG_NOTICE );
    updateSwitchIfChanged( m_indiP_zeroDm, "request", pcf::IndiElement::Off, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_releaseTT )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_releaseTT, ipRecv );
    if( !ipRecv.find( "request" ) || ipRecv["request"].getSwitchState() != pcf::IndiElement::On )
    {
        return 0;
    }

    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    stopPrediction();

    // Operator override: the zero is still retried, but the main loop takes tip/tilt back on the next WFS frame.
    m_modalEngaged.store( false );
    log<text_log>( std::string( "Main-loop tip/tilt returned by operator" ) +
                       ( m_output.status().m_zeroConfirmed ? "" : " before the DDSPC zero was confirmed" ),
                   logPrio::LOG_NOTICE );
    updateSwitchIfChanged( m_indiP_releaseTT, "request", pcf::IndiElement::Off, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_mode )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_mode, ipRecv );
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    if( !removed() )
        return -1;
    unsigned          selected = 0;
    DDSPC::ShadowMode mode     = DDSPC::ShadowMode::Standard;
    for( const auto &entry : ipRecv.getElements() )
    {
        if( entry.first != "standard" && entry.first != "accel_assisted" )
            return -1;
        if( entry.second.getSwitchState() == pcf::IndiElement::On )
        {
            ++selected;
            mode = entry.first == "standard" ? DDSPC::ShadowMode::Standard : DDSPC::ShadowMode::AccelAssisted;
        }
        else if( entry.second.getSwitchState() != pcf::IndiElement::Off )
            return -1;
    }
    if( selected != 1 )
        return -1;
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_mode.store( mode );
    m_predicting = false;
    applyTraining();
    updateSwitchIfChanged( m_indiP_mode,
                           "standard",
                           mode == DDSPC::ShadowMode::Standard ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           INDI_OK );
    updateSwitchIfChanged( m_indiP_mode,
                           "accel_assisted",
                           mode == DDSPC::ShadowMode::AccelAssisted ? pcf::IndiElement::On : pcf::IndiElement::Off,
                           INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_gain )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_gain, ipRecv );
    if( !ipRecv.find( "target" ) )
        return -1;
    float gain;
    try
    {
        gain = number( ipRecv["target"] );
    }
    catch( const std::exception & )
    {
        return -1;
    }
    if( !std::isfinite( gain ) || std::abs( gain ) > 10 )
        return -1;
    std::lock_guard<std::mutex> indiLock( m_indiMutex );
    std::lock_guard<std::mutex> modelLock( m_modelMutex );
    m_gain = gain;
    updatesIfChanged<float>( m_indiP_gain, { "current", "target" }, { gain, gain }, INDI_OK );
    return 0;
}

inline INDI_NEWCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_exploration )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_exploration, ipRecv );
    if( !ipRecv.find( "target" ) )
        return -1;
    try
    {
        const auto                  text   = ipRecv["target"].get<std::string>();
        auto                        stages = parseExploration( text );
        std::lock_guard<std::mutex> indiLock( m_indiMutex );
        std::lock_guard<std::mutex> modelLock( m_modelMutex );
        m_exploration = std::move( stages );
        updatesIfChanged<std::string>( m_indiP_exploration, { "current", "target" }, { text, text }, INDI_OK );
        return 0;
    }
    catch( const std::exception & )
    {
        return -1;
    }
}

inline INDI_SETCALLBACK_DEFN( loPredCtrlAcc_operational, m_indiP_sync )( const pcf::IndiProperty &ipRecv )
{
    INDI_VALIDATE_CALLBACK_PROPS( m_indiP_sync, ipRecv );
    const bool  adc     = m_syncProperty == "ddspcTiming";
    const char *enabled = adc ? "enabled" : "mode_code", *locked = adc ? "valid" : "delay_lock";
    auto        value = [&]( const char *name )
    {
        try
        {
            return ipRecv.find( name ) ? number( ipRecv[name] ) : std::nan( "" );
        }
        catch( const std::exception & )
        {
            return std::nan( "" );
        }
    };
    const double mode = value( enabled ), lock = value( locked );
    m_upstreamEnabled.store( std::isfinite( mode ) ? int( mode ) : -1 );
    m_upstreamLocked.store( std::isfinite( lock ) ? int( lock ) : -1 );
    m_upstreamPhaseUs.store( value( "delay_phase_error_us" ) );
    m_upstreamLatencyUs.store( value( "read_latency_error_us" ) );
    m_upstreamNs.store( synco::monotonicNs() );
    const bool valid = mode == 1 && lock == 1;
    m_syncLocked.store( valid );
    if( !valid )
        m_assistedValid.store( false );
    return 0;
}

inline double loPredCtrlAcc_operational::number( const pcf::IndiElement &element )
{
    const auto   text = element.getValue();
    std::size_t  used;
    const double value = std::stod( text, &used );
    if( used != text.size() || !std::isfinite( value ) )
        throw std::invalid_argument( "Invalid finite INDI number" );
    return value;
}
} // namespace MagAOX::app
#endif
