/** \file synchronized_inputs.hpp
 * \brief Build counter-synchronized native input packets without clock synchronization or invented exposures.
 */
#ifndef PC_SYNCHRONIZED_INPUTS_HPP
#define PC_SYNCHRONIZED_INPUTS_HPP
#include "multirate_shadow.hpp"

namespace DDSPC
{
/// Observed state of the existing upstream semaphore/delay controller; a configured expectation is not this evidence.
struct UpstreamStatus
{
    std::uint64_t m_epoch{ 0 }; ///< Reviewed source/status incarnation; changes after restart or re-anchoring.
    bool          m_synchronized{ false }, m_locked{ false }; ///< Actual producer mode and current lock indication.
    std::int64_t  m_receivedNs{ -1 }; ///< Local status receipt, never converted into an acquisition timestamp.
};

/// Original complete request identity plus projection into the independently calibrated WFS modal basis.
struct ProjectedRequest
{
    NativeObservation    m_source;   ///< Original metadata/hash, without claiming physical application.
    std::array<float, 2> m_values{}; ///< Calibrated modal requests, not arbitrary DM image pixels.
};

/// Reproduce NativeObserver's FLOAT-bit diagnostic digest; it checks copy identity, not physical application.
std::uint64_t command_payload_hash( std::span<const float> values /**< [in] Entire coherent request snapshot. */ );

/// Form a WFS origin with local availability and expected counter grouping; the engine checks all actual ADC records.
bool synchronized_origin( const MultirateContract &contract /**< [in] Reviewed upstream/counter/modal convention. */,
                          const UpstreamStatus    &status /**< [in] Actual producer status observation. */,
                          const NativeObservation &modal /**< [in] Original calibrated WFS snapshot. */,
                          std::int64_t  decisionNs /**< [in] Actual frozen local data-availability cutoff. */,
                          std::int64_t  predictionDeadlineNs /**< [in] Explicit diagnostic deadline, -1 if unknown. */,
                          OnlineOrigin &record /**< [out] Owned origin; no exposure or camera parent invented. */ );

/// Project a complete coherent request snapshot using an explicitly supplied calibrated linear map.
bool project_request( const MultirateContract &contract /**< [in] Source/layout/unit convention. */,
                      const NativeObservation &source /**< [in] Original complete combined-request metadata. */,
                      std::span<const float>   values /**< [in] Entire immutable request image. */,
                      const Matrix &projection /**< [in] Modes by image-elements calibration, including sign/units. */,
                      ProjectedRequest &record /**< [out] Original identity and modal values. */ );

/// Sample the complete observed request timeline at actual supplied slot cutoffs; never assume physical holds.
bool synchronized_commands(
    const MultirateContract           &contract /**< [in] Declared request-domain convention. */,
    std::uint64_t                      wfsCounter /**< [in] Frame whose four input slots have closed. */,
    std::uint64_t                      sequence /**< [in] Independent adapter record counter. */,
    std::span<const ProjectedRequest>  history /**< [in] Complete oldest-first snapshots. */,
    const std::array<std::int64_t, 4> &sampledNs /**< [in] Actual sampler/slot cutoffs, not assumed slack. */,
    const NativeObservation           &closedBy /**< [in] Actual next modal frame closes this input record. */,
    OnlineCommandFrame                &record /**< [out] Complete observed request frame. */ );
} // namespace DDSPC
#endif
