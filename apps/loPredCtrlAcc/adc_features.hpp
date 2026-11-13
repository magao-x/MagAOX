/** \file adc_features.hpp
 * \brief Platform-independent causal ADC integration with explicit source identity and frozen deadlines.
 */
#ifndef PCADC_FEATURES_HPP
#define PCADC_FEATURES_HPP
#include "adc_input.hpp"
#include "utils.hpp"
#include <span>
namespace DDSPC
{
/// Window qualification outcome, independent of the transport's much looser health threshold.
enum class FeatureReason
{
    Ready,
    Incomplete,
    Gap,
    Stale,
    Invalid
};

/// Return a stable CSV description of a feature-window outcome.
const char *feature_reason( FeatureReason reason /**< [in] Window outcome. */ );

/// Strict decision-time requirements for a causal, zero-order-held acceleration integral.
struct FeatureConfig
{
    std::int64_t m_windowNs{ 500000 }; ///< Physical half-open window duration, independent of sample count.
    std::int64_t m_lagNs{ 0 };         ///< Window end precedes the decision by this explicit latency allowance.
    std::int64_t m_maxGapNs{ 175000 }; ///< Maximum held-sample age, including both window boundaries.
    std::int64_t m_maxAgeNs{ 500000 }; ///< Maximum decision age of the weighted acquisition timestamp.
};

/// Owned aggregate; invalid windows never expose a usable acceleration vector.
struct AccelFeature
{
    DDSPC::Matrix m_value{ DDSPC::Matrix::Zero( 2, 1 ) }; ///< Time-weighted decoded SI acceleration.
    FeatureReason m_reason{ FeatureReason::Incomplete };  ///< Qualification failure or Ready.
    std::int64_t  m_effectiveNs{ -1 }, m_latestNs{ -1 };  ///< Weighted acquisition time and newest contributing sample.
    std::int64_t  m_availableNs{ -1 }; ///< Latest contributing receipt time, never after the frozen decision.
    std::uint64_t m_generation{ 0 }, m_segment{ 0 }; ///< Source incarnation of all contributing samples.
    std::size_t   m_samples{ 0 }; ///< Contributing samples, including a held sample at the left boundary.
};

/// Integrate only samples available by the decision; never interpolate from a future/late sample.
AccelFeature acceleration_window( std::span<const AdcSample> samples /**< [in] Oldest-first retained data. */,
                                  std::int64_t               decisionNs /**< [in] Frozen common-clock WFS deadline. */,
                                  const FeatureConfig &config /**< [in] Explicit window, coverage and freshness. */ );

/// Fill preallocated aggregate storage without allocation; failure clears values and qualification.
void acceleration_window( std::span<const AdcSample> samples /**< [in] Oldest-first retained data. */,
                          std::int64_t               decisionNs /**< [in] Frozen common-clock deadline. */,
                          const FeatureConfig       &config /**< [in] Coverage and age policy. */,
                          AccelFeature              &result /**< [out] Preallocated two-channel result. */ );

} // namespace DDSPC
#endif
