/** \file phase7_adapter.cpp
 * \brief Execute the committed Phase 7 model code under separate symbols without reimplementing its mathematics.
 */
#include "phase7_replay.hpp"
#define DDSPC DDSPCPhase7
#include "phase7/shadow_controller.hpp"
#undef DDSPC

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Own the actual historical pair and only translate test-interface values.
class Phase7Adapter : public Phase6Replay
{
    std::unique_ptr<DDSPCPhase7::ShadowController> m_pair; ///< Pinned, separately compiled learner pair.

  public:
    /// Set the same fixed benchmark configuration as the optimized manager.
    Phase7Adapter( int   modes /**< [in] Controlled modes. */,
                   bool  qrd /**< [in] Learner selector. */,
                   float gamma /**< [in] Forgetting factor. */,
                   int   history /**< [in] WFS history. */,
                   int   future /**< [in] Horizon. */,
                   int   accelHistory /**< [in] Sensor history. */ );
    /// Forward a synthetic record with identical common histories.
    bool process( std::uint64_t          index /**< [in] Identity. */,
                  const Eigen::MatrixXf &y /**< [in] Residual. */,
                  const Eigen::MatrixXf &u /**< [in] Applied input. */,
                  const Eigen::MatrixXf &a /**< [in] Features. */,
                  bool                   timingValid /**< [in] Valid timing. */,
                  bool                   accelValid /**< [in] Valid sensor. */ ) override;
    /// Forward adaptation control.
    void training( bool assisted /**< [in] Model. */, bool enabled /**< [in] Adapt. */ ) override;
    /// Forward independent reset.
    void reset( bool assisted /**< [in] Model. */ ) override;
    /// Forward readiness-gated selection.
    bool select( bool assisted /**< [in] Model. */ ) override;
    /// Capture the baseline public observables.
    Phase6Status inspect( bool assisted /**< [in] Model. */ ) override;
};

Phase7Adapter::Phase7Adapter( int modes, bool qrd, float gamma, int history, int future, int accelHistory )
{
    DDSPCPhase7::ShadowConfig cfg;
    cfg.m_modes        = modes;
    cfg.m_useQrd       = qrd;
    cfg.m_gamma        = gamma;
    cfg.m_history      = history;
    cfg.m_future       = future;
    cfg.m_accelHistory = accelHistory;
    cfg.m_prior        = .1f;
    cfg.m_lambda       = .3f;
    m_pair             = std::make_unique<DDSPCPhase7::ShadowController>( cfg );
}
bool Phase7Adapter::process( std::uint64_t          index,
                             const Eigen::MatrixXf &y,
                             const Eigen::MatrixXf &u,
                             const Eigen::MatrixXf &a,
                             bool                   timingValid,
                             bool                   accelValid )
{
    DDSPCPhase7::ShadowRecord record;
    record.m_index          = index;
    record.m_residual       = y;
    record.m_appliedCommand = u;
    record.m_acceleration   = a;
    record.m_wfsValid = record.m_appliedValid = true;
    record.m_timingValid                      = timingValid;
    record.m_accelValid                       = accelValid;
    return m_pair->process( record );
}
void Phase7Adapter::training( bool assisted, bool enabled )
{
    m_pair->training( assisted ? DDSPCPhase7::ShadowMode::AccelAssisted : DDSPCPhase7::ShadowMode::Standard, enabled );
}
void Phase7Adapter::reset( bool assisted )
{
    m_pair->reset( assisted ? DDSPCPhase7::ShadowMode::AccelAssisted : DDSPCPhase7::ShadowMode::Standard );
}
bool Phase7Adapter::select( bool assisted )
{
    return m_pair->select( assisted ? DDSPCPhase7::ShadowMode::AccelAssisted : DDSPCPhase7::ShadowMode::Standard );
}
Phase6Status Phase7Adapter::inspect( bool assisted )
{
    auto mode = assisted ? DDSPCPhase7::ShadowMode::AccelAssisted : DDSPCPhase7::ShadowMode::Standard;
    auto s    = m_pair->status( mode );
    return { { s.m_ready, s.m_trainingEnabled, s.m_training, s.m_dataValid, s.m_predictionValid },
             { s.m_samples, s.m_updates, s.m_predictions },
             { s.m_features, s.m_rank, static_cast<int>( s.m_reason ) },
             s.m_rmse,
             m_pair->prediction_matrix( mode ),
             m_pair->proposal( mode ) };
}
std::unique_ptr<Phase6Replay>
phase7_replay( int modes, bool qrd, float gamma, int history, int future, int accelHistory )
{
    return std::make_unique<Phase7Adapter>( modes, qrd, gamma, history, future, accelHistory );
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
