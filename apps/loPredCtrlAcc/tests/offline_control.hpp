/** \file offline_control.hpp
 * \brief Compatibility names delegating the offline simulation to shared production-independent components.
 * \ingroup loPredCtrlAcc_unit_test
 */
#ifndef loPredCtrlAcc_offline_control_hpp
#define loPredCtrlAcc_offline_control_hpp
#include "../shadow_policy.hpp"
/// \cond OFFLINE_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
using DDSPC::acceleration_window;
using DDSPC::AccelFeature;
using DDSPC::bounded_command;
using DDSPC::ControlDecision;
using DDSPC::ControlEvidence;
using DDSPC::feature_reason;
using DDSPC::FeatureConfig;
using DDSPC::FeatureReason;
using DDSPC::PolicyConfig;
using OfflineMode   = DDSPC::PolicyMode;
using OfflinePolicy = DDSPC::ShadowPolicy;

/// Preserve the simulation CSV terminology without duplicating policy logic.
const char *offline_mode( OfflineMode mode /**< [in] Simulated effective policy. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
