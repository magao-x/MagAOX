/** \file confirmed_replay.hpp
 * \brief Passive recording from the existing accelSim model, with independently consumed synthetic inputs.
 */
#ifndef loPredCtrlAcc_confirmed_replay_hpp
#define loPredCtrlAcc_confirmed_replay_hpp
#include "adc_forwarder.hpp"

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// One immutable virtual acquisition; it is never derived from a controller recommendation.
struct RecordedAdc
{
    DDSPC::AdcFrame m_frame;            ///< Original UINT16 payload and synthetic realtime timestamps.
    std::int64_t    m_receivedNs{ 0 };  ///< Known common-monotonic decoder availability.
    bool            m_restart{ false }; ///< Source replacement at this acquisition.
};

/// All forcing, WFS noise and actual limited inputs exist before the first application callback.
struct ConfirmedRecording
{
    DDSPC::InputConfig       m_inputs; ///< Explicit synthetic source, mode, response-delay and clock contract.
    std::vector<RecordedAdc> m_adc;    ///< 8-kHz acquisitions from actual accelSimDetail::model.
    std::vector<DDSPC::TimedWfsRecord>     m_wfs;   ///< 2-kHz exposure averages plus the independently consumed input.
    std::vector<DDSPC::AppliedInputRecord> m_holds; ///< Closed actual holds, with independently known effective times.
    std::vector<DDSPC::Matrix> m_requested; ///< Pre-limit requests retained only for an independent provenance check.
    double                     m_arcsecPerUnit{
        0 }; ///< Fixed 28-Hz model RMS used to normalize WFS/input units, not fitted from future data.
};

/// Replay faults alter observations/provenance, leaving the immutable physical recording unchanged.
struct ConfirmedReplayOptions
{
    std::size_t m_skipForwardStart{ 0 }, m_skipForwardCount{ 0 }; ///< Pause forwarding by WFS index, not acquisition.
    std::size_t m_skipWfsStart{ 0 }, m_skipWfsCount{ 0 }; ///< Omit WFS records while ADC and actual inputs continue.
    std::size_t m_badHoldStart{ 0 }, m_badHoldCount{ 0 }; ///< Substitute DriverPrepared for confirmed input evidence.
    std::size_t m_adcChangeStart{ 0 }; ///< First ADC vector index to perturb for sensor-only negative controls.
    int         m_adcOffsetCounts{
        0 }; ///< Add this bounded code offset after m_adcChangeStart; WFS and applied inputs unchanged.
    std::int64_t m_deliveryDelayNs{ 0 };   ///< Extra ADC receipt latency, with acquisition/publication times unchanged.
    std::size_t  m_captureCapacity{ 128 }; ///< Actual bounded capture retention, independent of source depth128.
    bool         m_wrongClock{ false };    ///< Negative control: ADC provenance has an unaccepted mapping epoch.
};

/// Passive execution evidence, without claims about closed-loop suppression or native scheduling.
struct ConfirmedReplayResult
{
    ForwardingStatus    m_forwarding; ///< Original capture and consumer loss/identity evidence.
    std::uint64_t       m_wfsAccepted{ 0 }, m_holdsAccepted{ 0 }; ///< Actual application admission counts.
    std::vector<double> m_frameUs; ///< ADC drain/forwarding, confirmations and WFS processing per nominal 500-us frame.
};

/// Generate actual accelSim forcing, exposure averages and independent limited-input histories in memory only.
ConfirmedRecording record_accelsim( std::size_t   frames = 800 /**< [in] Number of 2-kHz WFS records. */,
                                    std::uint64_t seed   = 721 /**< [in] Fixed numerical seed; no runtime producer. */,
                                    std::size_t   restartTick = std::numeric_limits<std::size_t>::max()
                                    /**< [in] Optional ADC source replacement; physical forcing continues. */ );

/// Dispatch only already available records through actual capture, bounded forwarding and application seams.
ConfirmedReplayResult
replay_confirmed( ReplaySink                   &sink /**< [in,out] Actual app input-only replay integration. */,
                  const ConfirmedRecording     &recording /**< [in] Independent immutable evidence. */,
                  const ConfirmedReplayOptions &options = {} /**< [in] Replay-only loss/provenance controls. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
