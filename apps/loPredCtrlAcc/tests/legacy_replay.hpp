/** \file legacy_replay.hpp
 * \brief Stream-free adapters for executing pinned historical and current legacy code on identical inputs.
 */
#ifndef loPredCtrlAcc_legacy_replay_hpp
#define loPredCtrlAcc_legacy_replay_hpp

#include "app_test_support.hpp"
#include <Eigen/Dense>
#include <memory>
#include <utility>
#include <stdexcept>

/// \cond LEGACY_TEST_SUPPORT
namespace libXWCTest::loPredCtrlAccTest
{
using ReplayMatrix = Eigen::MatrixXf;

/// Internal QR state exposed by the historical learner's public fields, without private-member rewriting.
struct LegacyQrSnapshot
{
    ReplayMatrix m_factor, m_augmented, m_gain, m_theta; ///< Factor, eliminated workspace, gain and weights.
};

/// Direct historical QR seam, independent of controller histories and command calculations.
class LegacyQr
{
  public:
    /// Release the independently owned historical learner.
    virtual ~LegacyQr() = default;

    /// Apply only the historical adjacent-row Givens elimination.
    virtual ReplayMatrix rotate( const ReplayMatrix &workspace /**< [in] Augmented (p+1)-by-p matrix. */ ) = 0;

    /// Learn one explicit regressor/target pair and copy the public numerical state.
    virtual LegacyQrSnapshot update( ReplayMatrix x /**< [in] Feature column. */,
                                     ReplayMatrix y /**< [in] Target column. */ ) = 0;
};

/// Adapter instantiated only where the actual pinned historical namespace is visible.
template <class Qr>
class LegacyQrAdapter : public LegacyQr
{
    Qr m_model; ///< Solely owned unmodified historical learner.

  public:
    /// Forward the original fixed dimensions and initialization.
    LegacyQrAdapter( int   outputs /**< [in] Target rows. */,
                     int   features /**< [in] Regressor rows. */,
                     float gamma /**< [in] Forgetting factor. */,
                     float prior /**< [in] Prior information strength. */ );

    /// Rotate a common explicit workspace.
    ReplayMatrix rotate( const ReplayMatrix &workspace /**< [in] Augmented matrix. */ ) override;

    /// Forward an observation and return all relevant post-update state.
    LegacyQrSnapshot update( ReplayMatrix x /**< [in] Regressor. */, ReplayMatrix y /**< [in] Target. */ ) override;
};

template <class Qr>
LegacyQrAdapter<Qr>::LegacyQrAdapter( int outputs, int features, float gamma, float prior )
    : m_model( outputs, features, gamma, prior )
{
}

template <class Qr>
ReplayMatrix LegacyQrAdapter<Qr>::rotate( const ReplayMatrix &workspace )
{
    m_model.A_ = workspace;
    m_model.apply_givens();
    return m_model.A_;
}

template <class Qr>
LegacyQrSnapshot LegacyQrAdapter<Qr>::update( ReplayMatrix x, ReplayMatrix y )
{
    m_model.update( &x, &y );
    return { m_model.R_, m_model.A_, m_model.g, m_model.prediction_matrix };
}

/// Construct the actual pinned historical QR learner, with separate symbols from production code.
std::unique_ptr<LegacyQr> original_qr( int   outputs /**< [in] Target rows. */,
                                       int   features /**< [in] Regressor rows. */,
                                       float gamma /**< [in] Forgetting factor. */,
                                       float prior /**< [in] Information strength. */ );

/// Identical numerical inputs for historical, restored, and accelerometer-disabled legacy models.
struct LegacySettings
{
    int   m_modes{ 1 }, m_history{ 5 }, m_future{ 3 }; ///< Modal dimensions and history/horizon lengths.
    float m_gain{ 0 }, m_gamma{ 0.99f }, m_lambda{ 100 }, m_prior{ 1 }; ///< Original operational gains/priors.
    bool  m_qrd{ false };                                               ///< Algorithm explicitly matched across models.
};

/// Observable model state through original public APIs, without private-member rewriting.
struct LegacySnapshot
{
    ReplayMatrix m_futureY, m_pastY, m_futureU, m_pastU, m_currentY, m_currentU,
        m_theta; ///< Public history/fit snapshots.
};

/// Type-erased historical/current numerical interface; all data stay in host memory.
class LegacyModel
{
  public:
    /// Destroy the independently owned model.
    virtual ~LegacyModel() = default;

    /// Generate and internally record a legacy command using the original exploration interface.
    virtual ReplayMatrix command( const ReplayMatrix &y /**< [in] WFS residual. */,
                                  const ReplayMatrix &noise /**< [in] Exploration. */ ) = 0;

    /// Update the legacy learner, optionally recomputing its operational controller.
    virtual void learn( bool solve /**< [in] Recompute the original scaled controller. */ ) = 0;

    /// Reset buffers alone or the entire model through original APIs.
    virtual void reset( bool buffersOnly /**< [in] Keep learned weights/controller. */ ) = 0;

    /// Switch the legacy regularization buffer.
    virtual void regularization( float value /**< [in] Scaled legacy penalty. */ ) = 0;

    /// Return copies of the histories and prediction matrix.
    virtual LegacySnapshot snapshot() = 0;

    /// Save legacy numerical state to a test-owned path only.
    virtual void save( const std::string &path /**< [in] Temporary test artifact prefix. */ ) = 0;

    /// Load a same-dimension legacy numerical artifact.
    virtual void load( const std::string &path /**< [in] Temporary test artifact prefix. */ ) = 0;
};

/// Extract histories only through methods that existed in the original controller.
template <class Model>
LegacySnapshot legacy_snapshot( Model &model /**< [in] Historical or current legacy instance. */,
                                int    history /**< [in] WFS history length. */ )
{
    return { model.get_measurement_future(),
             model.get_measurement_past(),
             model.get_command_future( 1 ),
             model.get_command_past(),
             model.get_current_measurement_past( history ),
             model.get_current_command_past( history ),
             model.get_prediction_matrix() };
}

/// Reusable numerical wrapper instantiated with distinct historical/restored/current namespace types.
template <class Model>
class LegacyModelAdapter : public LegacyModel
{
    Model m_model;   ///< Owned model; namespaces keep historical/current symbols independent.
    int   m_history; ///< Original history size for public state snapshots.
  public:
    /// Forward matched settings and any current-only zero-accelerometer constructor parameters.
    template <class... Extra>
    LegacyModelAdapter( const LegacySettings &s /**< [in] Matched parameters. */,
                        Extra... extra /**< [in] Current-only arguments. */ );

    /// Forward the original command calculation.
    ReplayMatrix command( const ReplayMatrix &y /**< [in] Residual. */,
                          const ReplayMatrix &noise /**< [in] Exploration. */ ) override;

    /// Forward original identification and optional controller solve.
    void learn( bool solve /**< [in] Compute operational gain. */ ) override;

    /// Forward the requested reset.
    void reset( bool buffersOnly /**< [in] Reset histories only. */ ) override;

    /// Forward legacy regularization switching.
    void regularization( float value /**< [in] Scaled penalty. */ ) override;

    /// Copy public history and coefficient state.
    LegacySnapshot snapshot() override;

    /// Forward legacy numerical persistence to a test-owned path.
    void save( const std::string &path /**< [in] Test artifact prefix. */ ) override;

    /// Forward a same-dimension artifact restore.
    void load( const std::string &path /**< [in] Test artifact prefix. */ ) override;
};

template <class Model>
template <class... Extra>
LegacyModelAdapter<Model>::LegacyModelAdapter( const LegacySettings &s, Extra... extra )
    : m_model( s.m_modes, s.m_history, s.m_future, s.m_gain, s.m_gamma, s.m_lambda, s.m_prior, extra... ),
      m_history( s.m_history )
{
    m_model.use_qrd = s.m_qrd;
}

template <class Model>
ReplayMatrix LegacyModelAdapter<Model>::command( const ReplayMatrix &y, const ReplayMatrix &noise )
{
    return m_model.calculate_command( y, noise );
}

template <class Model>
void LegacyModelAdapter<Model>::learn( bool solve )
{
    m_model.update_system();
    if( solve )
        m_model.update_controller();
}

template <class Model>
void LegacyModelAdapter<Model>::reset( bool buffersOnly )
{
    if( buffersOnly )
        m_model.reset_buffers();
    else
        m_model.reset();
}

template <class Model>
void LegacyModelAdapter<Model>::regularization( float value )
{
    m_model.set_regularization( value );
}

template <class Model>
LegacySnapshot LegacyModelAdapter<Model>::snapshot()
{
    return legacy_snapshot( m_model, m_history );
}

template <class Model>
void LegacyModelAdapter<Model>::save( const std::string &path )
{
    m_model.save_state( path );
}

template <class Model>
void LegacyModelAdapter<Model>::load( const std::string &path )
{
    m_model.load_state( path );
}

/// Application observations after a callback and optional mock publication, with no stream connection.
struct LegacyAppFrame
{
    ReplayMatrix m_candidate, m_output,
        m_published;        ///< Calculated delta, accumulated buffer and last mock publication.
    LegacySnapshot m_model; ///< Histories/identification after this WFS callback.
};

/// Configuration/state distinctions exposed by the real app headers through protected test harness access.
struct LegacyAppState
{
    float m_gain{ 0 }, m_copyGain{ 0 }; ///< Feedback and nonpredictive copy gains.
    bool  m_qrd{ false }, m_predict{ false }, m_learn{ false }, m_integrate{ false }, m_shadow{ false },
        m_accel{ false }, m_shadowAllocated{ false }; ///< Operational and shadow state.
};

/// Stream-free app replay interface; initialization seeds only the otherwise undefined output buffer.
class LegacyApp
{
  public:
    /// Shut down only offline lifecycle doubles and delete the model.
    virtual ~LegacyApp() = default;

    /// Load actual app config/startup/allocation with mocked bases and known initial output zero.
    virtual int initialize( const std::map<std::string, std::string> &config /**< [in] Identical fixture settings. */,
                            int                                       width /**< [in] WFS/output width. */ ) = 0;

    /// Run the actual WFS callback and optionally copy to a host-memory publication sink.
    virtual LegacyAppFrame frame( ReplayMatrix y /**< [in] Residual payload. */,
                                  bool         publish /**< [in] Perform mock publication. */ ) = 0;

    /// Send a real app callback request for switches or the exploration sequence.
    virtual int control( const std::string &name /**< [in] Property. */,
                         const std::string &value /**< [in] Request. */ ) = 0;

    /// Inspect configuration and execution gates.
    virtual LegacyAppState state_snapshot() = 0;
};

/// Shared protected-access harness for original/restored and current app member naming conventions.
template <class App, bool Acc>
class LegacyAppAdapter : public App, public LegacyApp
{
    bool         m_started{ false }; ///< This fixture started its lifecycle doubles.
    ReplayMatrix m_published;        ///< Last host-memory mock publication; a skipped copy models coalescing/hold.
    /// Return the actual app's legacy model pointer.
    auto *model();

  public:
    /// Shutdown the real app body against doubles; no native helper exists in this translation unit.
    ~LegacyAppAdapter() override;

    /// Initialize the real application header with stream-free lifecycle doubles.
    int initialize( const std::map<std::string, std::string> &config /**< [in] Fixture settings. */,
                    int                                       width /**< [in] Payload width. */ ) override;

    /// Run the actual callback and optional copy-to-memory method.
    LegacyAppFrame frame( ReplayMatrix y /**< [in] WFS payload. */,
                          bool         publish /**< [in] Mock publisher schedule. */ ) override;

    /// Dispatch matching property callbacks, including rejected shadow selections in legacy execution.
    int control( const std::string &name /**< [in] Property name. */,
                 const std::string &value /**< [in] Requested value. */ ) override;

    /// Observe legacy parameters and the separation from shadow state.
    LegacyAppState state_snapshot() override;
};

template <class App, bool Acc>
auto *LegacyAppAdapter<App, Acc>::model()
{
    if constexpr( Acc )
        return this->m_controller;
    else
        return this->controller;
}

template <class App, bool Acc>
LegacyAppAdapter<App, Acc>::~LegacyAppAdapter()
{
    if( m_started )
        this->appShutdown();
}

template <class App, bool Acc>
int LegacyAppAdapter<App, Acc>::initialize( const std::map<std::string, std::string> &config, int width )
{
    this->setupConfig();
    this->config.m_values = config;
    if( this->loadConfigImpl( this->config ) < 0 || this->appStartup() < 0 )
        return -1;
    m_started       = true;
    using Wfs       = MagAOX::app::dev::shmimMonitor<App>;
    Wfs::m_width    = width;
    Wfs::m_height   = 1;
    Wfs::m_dataType = _DATATYPE_FLOAT;
    if( this->allocate( MagAOX::app::dev::shmimT{} ) < 0 )
        return -1;
    // Original allocation leaves this uninitialized. Define the same test precondition for all three paths.
    if constexpr( Acc )
        this->m_fullCommand.setZero();
    else
        this->full_command.setZero();
    m_published = ReplayMatrix::Zero( width, 1 );
    return this->configureAcquisition();
}

template <class App, bool Acc>
LegacyAppFrame LegacyAppAdapter<App, Acc>::frame( ReplayMatrix y, bool publish )
{
    if( this->processImage( y.data(), MagAOX::app::dev::shmimT{} ) < 0 )
        throw std::runtime_error( "Legacy WFS callback failed" );
    if( publish && this->loadImageIntoStream( m_published.data() ) < 0 )
        throw std::runtime_error( "Mock publication failed" );
    LegacyAppFrame result;
    if constexpr( Acc )
    {
        result.m_candidate = this->m_newCommand;
        result.m_output    = this->m_fullCommand;
    }
    else
    {
        result.m_candidate = this->new_command;
        result.m_output    = this->full_command;
    }
    result.m_published = m_published;
    result.m_model     = legacy_snapshot( *model(), this->m_history );
    return result;
}

template <class App, bool Acc>
int LegacyAppAdapter<App, Acc>::control( const std::string &name, const std::string &value )
{
    auto invoke = [&]( pcf::IndiProperty &property, auto callback, const char *element )
    {
        auto request = property;
        if( request.getType() == pcf::IndiProperty::Text )
            request[element].set( value );
        else
            request[element].setSwitchState( value == "on" ? pcf::IndiElement::On : pcf::IndiElement::Off );
        return ( this->*callback )( request );
    };
    if( name == "predict" )
        return invoke( this->m_indiP_predictingToggle, &App::newCallBack_m_indiP_predictingToggle, "toggle" );
    if( name == "learn" )
        return invoke( this->m_indiP_learningToggle, &App::newCallBack_m_indiP_learningToggle, "toggle" );
    if( name == "learn_std" )
        return invoke( this->m_indiP_learningStdToggle, &App::newCallBack_m_indiP_learningStdToggle, "toggle" );
    if( name == "integrate" )
        return invoke( this->m_indiP_integratingToggle, &App::newCallBack_m_indiP_integratingToggle, "toggle" );
    if( name == "reset_model" )
        return invoke( this->m_indiP_resetToggle, &App::newCallBack_m_indiP_resetToggle, "request" );
    if( name == "exploration_sequence" )
        return invoke( this->m_indiP_exploration, &App::newCallBack_m_indiP_exploration, "target" );
    if constexpr( Acc )
    {
        if( name == "ddspc_mode" )
        {
            auto request = this->m_indiP_ddspcMode;
            for( const auto &entry : request.getElements() )
                request[entry.first].setSwitchState( entry.first == value ? pcf::IndiElement::On
                                                                          : pcf::IndiElement::Off );
            return this->newCallBack_m_indiP_ddspcMode( request );
        }
        if( name == "ddspc_train" )
            return invoke( this->m_indiP_ddspcTrain, &App::newCallBack_m_indiP_ddspcTrain, "standard" );
    }
    return -1;
}

template <class App, bool Acc>
LegacyAppState LegacyAppAdapter<App, Acc>::state_snapshot()
{
    LegacyAppState result;
    result.m_gain = this->m_gainCtrl;
    if constexpr( Acc )
    {
        result.m_copyGain        = this->m_copygainCtrl;
        result.m_qrd             = this->m_useQrd;
        result.m_predict         = this->m_isPredictiveControl;
        result.m_learn           = this->m_isLearning;
        result.m_integrate       = this->m_isIntegrating;
        result.m_shadow          = this->m_shadowMode;
        result.m_accel           = this->m_accelEnabled;
        result.m_shadowAllocated = bool( this->m_shadowController ) || bool( this->m_shadowInputs );
    }
    else
    {
        result.m_copyGain  = this->m_gainCtrl;
        result.m_qrd       = this->use_qrd;
        result.m_predict   = this->is_predictive_control;
        result.m_learn     = this->is_learning;
        result.m_integrate = this->is_integrating;
    }
    return result;
}

/// Construct the pinned pre-accelerometer numerical model.
std::unique_ptr<LegacyModel> original_model( const LegacySettings &settings /**< [in] Matched settings. */ );
/// Construct the unchanged restored loPredCtrl numerical model.
std::unique_ptr<LegacyModel> restored_model( const LegacySettings &settings /**< [in] Matched settings. */ );
/// Construct the current accelerometer-disabled Legacy-mode numerical model.
std::unique_ptr<LegacyModel> current_legacy_model( const LegacySettings &settings /**< [in] Matched settings. */ );
/// Construct the pinned original app header with offline bases.
std::unique_ptr<LegacyApp> original_app();
/// Construct the verified working-tree restored app header with offline bases.
std::unique_ptr<LegacyApp> restored_app();
/// Construct the current loPredCtrlAcc app header with offline bases.
std::unique_ptr<LegacyApp> current_legacy_app();

} // namespace libXWCTest::loPredCtrlAccTest
/// \endcond
#endif
