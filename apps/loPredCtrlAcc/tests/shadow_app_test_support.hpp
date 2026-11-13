/** \file shadow_app_test_support.hpp
 * \brief Real application header connected to input-only replay with explicit stream-free lifecycle doubles.
 */
#ifndef loPredCtrlAcc_shadow_app_test_support_hpp
#define loPredCtrlAcc_shadow_app_test_support_hpp
#include "app_test_support.hpp"
#include "../loPredCtrlAcc.hpp"
#include "shadow_replay.hpp"
#include <stdexcept>

/// \cond TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
/// Current frozen decision evidence, retained only by tests and never visible to the recording generator.
struct ReplayObservation
{
    DDSPC::ShadowPipelineStatus m_status;    ///< Actual app pipeline state after this WFS origin.
    DDSPC::Matrix               m_candidate; ///< Diagnostic copy, never passed to the applied-input replay interface.
};

/// Stream-free endpoint invoking the actual application's configuration, input seams, models and output guards.
class ReplayApplication : public MagAOX::app::loPredCtrlAcc, public ReplaySink
{
  public:
    bool          m_trace{ true }; ///< Disable observation allocations during timing measurements.
    std::uint64_t m_validCandidates{ 0 }, m_assistedCandidates{ 0 }; ///< Observed actual app eligibility counts.
    bool          m_forgeFeatures{ false };        ///< Negative control: prebuilt WFS acceleration must be ignored.
    std::vector<ReplayObservation> m_observations; ///< Frozen decision evidence, owned solely by the test.

    /// Configure through the actual loader/startup and establish only the explicit synthetic replay contract.
    void prepare( const DDSPC::ShadowConfig &models /**< [in] Model dimensions and numerical parameters. */,
                  const DDSPC::InputConfig  &inputs /**< [in] Independent synthetic timing/input contract. */,
                  bool                       enabled   = true /**< [in] Omit the new flag entirely when false. */,
                  DDSPC::PolicyMode          requested = DDSPC::PolicyMode::Assisted /**< [in] Diagnostic request. */,
                  double                     maxStep   = .2 /**< [in] Explicit test candidate slew bound. */ );

    /// Validate arbitrary startup values through the real loader, without starting any helper.
    int load_values( const std::map<std::string, std::string> &values /**< [in] Test-only startup values. */ );

    /// Verify unconditional guarded-build output denial even if the runtime shadow flag is inconsistent.
    bool bad_runtime_outputs_rejected();

    /// Enable adaptation through the real independent-training callback; preparation leaves training disabled.
    void train( bool standard = true /**< [in] Enable the WFS-only learner. */,
                bool assisted = true /**< [in] Enable the independent assisted learner. */ );

    /// Feed actual decoder output into the application's new shared pipeline.
    bool adc( const DDSPC::ShadowAdcRecord &record /**< [in] Decoded evidence. */,
              std::int64_t                  nowNs /**< [in] Virtual availability. */ ) override;

    /// Feed a frozen WFS origin through the actual app, optionally copying diagnostics after processing.
    bool wfs( const DDSPC::TimedWfsRecord &record /**< [in] Exposure/residual source. */,
              std::int64_t                 nowNs /**< [in] Origin time. */ ) override;

    /// Supply only an independent command certificate through the actual app's confirmed-input seam.
    bool applied( const DDSPC::AppliedInputRecord &record /**< [in] Held input evidence. */,
                  std::int64_t                     nowNs /**< [in] Confirmation time. */ ) override;

    /// Return actual learner evidence for regression comparisons, independent of candidate selection.
    DDSPC::Matrix model( DDSPC::ShadowMode mode /**< [in] Independent learner. */ );

    /// Return actual readiness/training state without fabricating application input validity.
    DDSPC::ShadowStatus model_status( DDSPC::ShadowMode mode /**< [in] Independent learner. */ );

    /// Check all Legacy/output-owning helper paths remained inactive.
    bool inhibited() const;

    /// Expose the actual expiry seam, without reading a wall clock or waiting for new measurements.
    bool expire( std::int64_t nowNs /**< [in] Virtual silence deadline. */ );

    /// Exercise direct output entrypoints; they must reject without changing the sentinel destination.
    bool outputs_rejected();

    /// Exercise a raw WFS callback, which must discard any explicitly synthetic timing session.
    void raw_wfs();

    using MagAOX::app::loPredCtrlAcc::shadowCandidate;
    using MagAOX::app::loPredCtrlAcc::shadowPipelineStatus;
};

/// Small strict-QR configuration matching the existing offline ADC/DDSPC baseline dimensions.
inline DDSPC::ShadowConfig replay_model_config()
{
    DDSPC::ShadowConfig c;
    c.m_modes = c.m_accelChannels = 2;
    c.m_history                   = 4;
    c.m_future = c.m_accelHistory = 3;
    c.m_gamma                     = .9995f;
    c.m_prior                     = .001f;
    c.m_lambda                    = .1f;
    c.m_useQrd                    = true;
    return c;
}

inline void ReplayApplication::prepare( const DDSPC::ShadowConfig &models,
                                        const DDSPC::InputConfig  &inputs,
                                        bool                       enabled,
                                        DDSPC::PolicyMode          requested,
                                        double                     maxStep )
{
    config.m_values = {
        { "parameters.shadow", "1" },
        { "parameters.accel_enabled", "0" },
        { "parameters.num_modes", std::to_string( models.m_modes ) },
        { "parameters.history", std::to_string( models.m_history ) },
        { "parameters.future", std::to_string( models.m_future ) },
        { "parameters.accel_channels", "2" },
        { "parameters.accel_history", std::to_string( models.m_accelHistory ) },
        { "parameters.gamma", std::to_string( models.m_gamma ) },
        { "parameters.covariance", std::to_string( models.m_prior ) },
        { "parameters.regularization", std::to_string( models.m_lambda ) },
        { "parameters.qrd", models.m_useQrd ? "1" : "0" },
        { "shadow_pipeline.requested", requested == DDSPC::PolicyMode::Assisted ? "accel_assisted" : "standard" },
        { "shadow_pipeline.command_limit", "3" },
        { "shadow_pipeline.max_step", std::to_string( maxStep ) } };
    if( enabled )
        config.m_values["parameters.shadow_pipeline"] = "1";
    setupConfig();
    if( loadConfigImpl( config ) != 0 || appStartup() != 0 || configureShadowInputs( inputs ) != 0 )
        throw std::runtime_error( "Offline application replay preparation failed" );
}

inline int ReplayApplication::load_values( const std::map<std::string, std::string> &values )
{
    config.m_values = values;
    setupConfig();
    return loadConfigImpl( config );
}

inline bool ReplayApplication::bad_runtime_outputs_rejected()
{
#ifdef DDSPC_SHADOW_ONLY
    const bool old  = m_shadowMode;
    m_shadowMode    = false;
    const bool safe = outputs_rejected();
    m_shadowMode    = old;
    return safe;
#else
    return true; // The ordinary build intentionally retains Legacy dispatch.
#endif
}

inline void ReplayApplication::train( bool standard, bool assisted )
{
    auto request = m_indiP_ddspcTrain;
    request["standard"].setSwitchState( standard ? pcf::IndiElement::On : pcf::IndiElement::Off );
    request["accel_assisted"].setSwitchState( assisted ? pcf::IndiElement::On : pcf::IndiElement::Off );
    if( newCallBack_m_indiP_ddspcTrain( request ) != 0 )
        throw std::runtime_error( "Replay training request failed" );
}

inline bool ReplayApplication::adc( const DDSPC::ShadowAdcRecord &record, std::int64_t nowNs )
{
    return processShadowAdc( record, nowNs );
}

inline bool ReplayApplication::wfs( const DDSPC::TimedWfsRecord &record, std::int64_t nowNs )
{
    bool accepted;
    if( m_forgeFeatures )
    {
        auto supplied              = record;
        supplied.m_acceleration    = DDSPC::Matrix::Constant( 7, 2, 100000 );
        supplied.m_accelValid      = true;
        supplied.m_accelAcquiredNs = supplied.m_accelAvailableNs = nowNs;
        accepted                                                 = processShadowWfs( supplied, nowNs );
    }
    else
        accepted = processShadowWfs( record, nowNs );
    const auto current = shadowPipelineStatus();
    m_validCandidates += current.m_candidateValid;
    m_assistedCandidates += current.m_candidateValid && current.m_effective == DDSPC::PolicyMode::Assisted;
    if( m_trace )
    {
        ReplayObservation observation;
        observation.m_status = shadowPipelineStatus();
        observation.m_candidate.setZero( record.m_residual.rows(), 1 );
        shadowCandidate( observation.m_candidate );
        m_observations.push_back( std::move( observation ) );
    }
    return accepted;
}

inline bool ReplayApplication::applied( const DDSPC::AppliedInputRecord &record, std::int64_t nowNs )
{
    return processShadowApplied( record, nowNs );
}

inline DDSPC::Matrix ReplayApplication::model( DDSPC::ShadowMode mode )
{
    return m_shadowController->prediction_matrix( mode );
}

inline DDSPC::ShadowStatus ReplayApplication::model_status( DDSPC::ShadowMode mode )
{
    return m_shadowController->status( mode );
}

inline bool ReplayApplication::inhibited() const
{
    return MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc>::m_loads == 0 &&
           MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc>::m_starts == 0 &&
           MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc>::m_logic == 0 &&
           MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc>::m_updates == 0 &&
           MagAOX::app::dev::frameGrabber<MagAOX::app::loPredCtrlAcc>::m_stops == 0 &&
           MagAOX::app::dev::telemeter<MagAOX::app::loPredCtrlAcc>::m_starts == 0 &&
           MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc, MagAOX::app::accelShmimT>::m_starts == 0 &&
           !m_controller && !m_updated && m_fullCommand.size() == 0 && !m_isLearning && !m_isStdLearning &&
           !m_isPredictiveControl && !m_isIntegrating && !m_doTriggerLoad && !m_doTriggerSave && !m_switchExploration;
}

inline bool ReplayApplication::expire( std::int64_t nowNs )
{
    return expireShadowInputs( nowNs );
}

inline bool ReplayApplication::outputs_rejected()
{
    float      sentinel[2]{ 123, 456 };
    const bool refused = configureAcquisition() < 0 && startAcquisition() < 0 && acquireAndCheckValid() != 0 &&
                         loadImageIntoStream( sentinel ) < 0;
    return refused && sentinel[0] == 123 && sentinel[1] == 456;
}

inline void ReplayApplication::raw_wfs()
{
    MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc>::m_width    = 2;
    MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc>::m_height   = 1;
    MagAOX::app::dev::shmimMonitor<MagAOX::app::loPredCtrlAcc>::m_dataType = _DATATYPE_FLOAT;
    float residual[2]{ 0, 0 };
    processImage( residual, MagAOX::app::dev::shmimT{} );
}
} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
