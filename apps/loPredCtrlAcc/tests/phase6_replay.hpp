/** \file phase6_replay.hpp
 * \brief Stream-free interface to the independently compiled, committed pre-optimization model manager.
 */
#ifndef loPredCtrlAcc_phase6_replay_hpp
#define loPredCtrlAcc_phase6_replay_hpp
#include <Eigen/Dense>
#include <array>
#include <cstdint>
#include <memory>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Exact discrete diagnostic fields plus numerical outputs, independent of either implementation's types.
struct Phase6Status
{
    std::array<bool, 5>          m_flags{};   ///< Ready, training-enabled, training, data-valid, prediction-valid.
    std::array<std::uint64_t, 3> m_counts{};  ///< Samples, updates and scored forecasts.
    std::array<int, 3>           m_codes{};   ///< Feature dimension, empirical rank and readiness reason.
    double                       m_rmse{ 0 }; ///< Frozen-origin conditional prediction error.
    Eigen::MatrixXf              m_theta, m_proposal; ///< Canonical fit and diagnostic input.
};

/// The pinned implementation is owned behind this interface and never links application or device code.
class Phase6Replay
{
  public:
    /// Release the independent baseline model pair.
    virtual ~Phase6Replay() = default;

    /// Feed the same explicitly synthetic immutable record as the optimized pair.
    virtual bool process( std::uint64_t          index /**< [in] External decision identity. */,
                          const Eigen::MatrixXf &y /**< [in] WFS residual. */,
                          const Eigen::MatrixXf &u /**< [in] Actual synthetic held input. */,
                          const Eigen::MatrixXf &a /**< [in] Synchronized sensor features. */,
                          bool                   timingValid /**< [in] Declared synthetic timing validity. */,
                          bool                   accelValid /**< [in] Sensor validity. */ ) = 0;

    /// Change adaptation independently of selection.
    virtual void training( bool assisted /**< [in] Select the assisted model. */,
                           bool enabled /**< [in] Adapt. */ ) = 0;

    /// Reset one model through the original public interface.
    virtual void reset( bool assisted /**< [in] Model to reset. */ ) = 0;

    /// Run the original readiness-gated diagnostic selection.
    virtual bool select( bool assisted /**< [in] Requested mode. */ ) = 0;

    /// Copy baseline status, coefficients and proposal.
    virtual Phase6Status inspect( bool assisted /**< [in] Model to inspect. */ ) = 0;
};

/// Construct the exact committed Phase 6 manager at the original benchmark history/horizon/sensor dimensions.
std::unique_ptr<Phase6Replay> phase6_replay( int   modes /**< [in] Controlled mode count. */,
                                             bool  qrd /**< [in] QR rather than conventional RLS. */,
                                             float gamma        = 1.f /**< [in] Per-update forgetting factor. */,
                                             int   history      = 5 /**< [in] WFS history steps. */,
                                             int   future       = 3 /**< [in] Future horizon. */,
                                             int   accelHistory = 20 /**< [in] Sensor history steps. */ );
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
