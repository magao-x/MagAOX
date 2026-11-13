/** \file shadow_replay.hpp
 * \brief Owned offline ADC/WFS/applied-input replay; producer records cannot observe shadow recommendations.
 * \ingroup loPredCtrlAcc_unit_test
 */
#ifndef loPredCtrlAcc_shadow_replay_hpp
#define loPredCtrlAcc_shadow_replay_hpp
#include "offline_adc.hpp"
#include "../shadow_pipeline.hpp"
#include <limits>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Input-only application seam; the replay generator has no candidate getter or command-output method.
class ReplaySink
{
  public:
    /// Release a replay endpoint without touching external resources.
    virtual ~ReplaySink() = default;

    /// Deliver actual AdcCapture output with its explicit synthetic clock/calibration provenance.
    virtual bool adc( const DDSPC::ShadowAdcRecord &record /**< [in] Decoded owned evidence. */,
                      std::int64_t                  nowNs /**< [in] Virtual receipt time. */ ) = 0;

    /// Deliver a measured WFS exposure at its frozen decision origin.
    virtual bool wfs( const DDSPC::TimedWfsRecord &record /**< [in] Owned WFS measurement. */,
                      std::int64_t                 nowNs /**< [in] Virtual deadline. */ ) = 0;

    /// Deliver an independently established closed input interval, never a shadow proposal.
    virtual bool applied( const DDSPC::AppliedInputRecord &record /**< [in] Owned hold certificate. */,
                          std::int64_t                     nowNs /**< [in] Confirmation receipt. */ ) = 0;
};

/// Immutable replay settings; commands/noise are generated independently before any application callback.
struct ShadowReplayConfig
{
    SyntheticAdcConfig m_adc;            ///< Actual encoder/decoder profile, jitter, delivery and observation faults.
    DDSPC::InputConfig m_inputs;         ///< Explicitly synthetic exposure, response and source/clock contract.
    std::size_t        m_frames{ 1200 }; ///< WFS observations at2 kHz; ADC nominal rate8 kHz.
    std::uint64_t      m_restartTick{ std::numeric_limits<std::uint64_t>::max() }; ///< ADC counter/generation restart.
    int  m_skipWfsStart{ 0 }, m_skipWfsCount{ 0 };       ///< Missing WFS observations; physical forcing still advances.
    int  m_badAppliedStart{ 0 }, m_badAppliedCount{ 0 }; ///< Driver-prepared intervals must be rejected by the app.
    bool m_wrongAdcClock{ false }; ///< Negative control: decoded source has an unaccepted clock mapping.
    std::size_t m_lateWfsIndex{ std::numeric_limits<std::size_t>::max() }; ///< Miss a frozen WFS origin deadline.
    bool        m_rejectEveryCommand{ false }; ///< Negative control: no confirmed applied input exists.
    double      m_wfsNoise{ .005 };            ///< Independent measured-WFS noise; not controller-generated.

    /// Establish a physically ordered synthetic exposure/confirmed-hold timeline in a known common clock.
    ShadowReplayConfig();
};

/// Per-callback elapsed work, including actual decoding and real application integration, without native pacing.
struct ShadowReplayResult
{
    std::vector<double> m_adcUs, m_wfsUs, m_commandUs, m_frameUs; ///< Actual local wall-clock processing durations.
    std::uint64_t       m_adcAccepted{ 0 }, m_wfsAccepted{ 0 }, m_commandsAccepted{ 0 }; ///< Admission evidence.
    std::size_t         m_eigenRequests{ 0 }, m_cppRequests{ 0 }; ///< Optional benchmark-only allocator observations.
    DDSPC::AdcStatus    m_capture; ///< Actual decoder loss, restart, sample-rate and validity evidence.
};

/// Generate owned records independently, then dispatch them in availability order through the application seam.
ShadowReplayResult replay_shadow( ReplaySink               &sink /**< [in,out] Actual application input integration. */,
                                  const ShadowReplayConfig &config /**< [in] Fixed deterministic recording. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
