/** \file learningCtrl.hpp
 * \brief PO4AO inference and optional managed training through the shared INDI publisher.
 */
#ifndef learningCtrl_hpp
#define learningCtrl_hpp

#include <cmath>
#include <map>
#include <set>
#include <string>

#include <QElapsedTimer>
#include <QInputDialog>
#include <QSignalBlocker>
#include <QTimer>

#include "ui_learningCtrl.h"
#include "../xWidgets/xWidget.hpp"

namespace xqt
{

/// Control PO4AO directly when its optional trainer is unavailable.
/** The shared publisher queues all callbacks onto this widget's Qt thread.
 * Managed compound operations belong to the companion. Without it, native C++
 * controls remain available and Python training stays entirely with the notebook.
 * No loop state or training state gates reconstruction controls.
 */
class learningCtrl : public xWidget
{
    Q_OBJECT

  protected:
    std::string m_device; ///< Companion INDI device, supplied by the launcher.

    std::string m_runtimeDevice; ///< C++ PO4AO device, available independently of the trainer.

    std::string m_runtimeInstance; ///< Process identity used to invalidate previous-runtime capabilities.

    QElapsedTimer m_runtimeAge; ///< Age since an advancing C++ model-status sequence.

    std::string m_runtimeSequence; ///< Last C++ sequence; empty until discovery.

    std::string m_nnDevice; ///< NN device supplied independently by the launcher.

    QElapsedTimer m_nnAge; ///< Age since the independent NN application's advancing heartbeat.

    std::string m_nnHeartbeat, m_nnInstance; ///< Current NN process-lifetime heartbeat baseline.

    bool m_connected{ false }; ///< Transport connection; readiness also needs property definitions.

    std::map<std::string, pcf::IndiProperty>
        m_properties; ///< Definitions and merged partial updates, owned by Qt thread.

    std::set<std::string> m_defined; ///< Properties with observed definitions, including command permissions.

    std::map<std::string, bool>
        m_pending; ///< Requests keyed by device.property; value records an observed Busy acknowledgement.

    bool m_managed{ false }; ///< Last selected command route, independent of transport connection.

    std::set<std::string> m_routePending; ///< Outstanding old-route commands, including other clients' Busy readbacks.

    bool m_nativeResetPending{ false }; ///< Native Reset completion is tracked by generations, not switch edges.

    qulonglong m_nativeResetAfter{ 0 }; ///< Last reset request generation before this window requested Reset.

    std::string m_message; ///< Local request/acknowledgement error, separate from backend status.

    QTimer m_statusTimer; ///< Re-evaluates expiry even when no INDI message arrives.

    QElapsedTimer m_heartbeatAge; ///< Monotonic age since an observed advancing companion heartbeat.

    std::string m_lastHeartbeat; ///< Last sequence, used to reject unchanged cached definitions.

    bool m_haveHeartbeat{ false }; ///< A baseline sequence has been observed in this device lifetime.

    QString m_policySelection; ///< Stable staged library ID, retained across catalog delete/redefine updates.

    QTimer m_ackTimer; ///< Bounds the wait for initial acknowledgement; never retries a command.

  public:
    /// Construct one panel for C++ inference and an optional companion.
    explicit learningCtrl( const std::string &device,                  /**< [in] Companion device name. */
                           QWidget           *parent        = nullptr, /**< [in] Optional Qt parent. */
                           const std::string &runtimeDevice = "po4ao", /**< [in] C++ PO4AO device. */
                           const std::string &nnDevice      = "nnReconstructor" /**< [in] Independent NN device. */ );

    /// Attach named property subscriptions through the existing publisher.
    void subscribe() override;

    /// Mark transport connection without inferring device readiness.
    void onConnect() override;

    /// Invalidate cached states and pending requests on disconnect.
    void onDisconnect() override;

    /// Replace a definition for its independently configured device.
    void
    handleDefProperty( const pcf::IndiProperty &property /**< [in] Definition received on the Qt thread. */ ) override;

    /// Merge sparse element updates without replacing the catalog or command permissions.
    void handleSetProperty( const pcf::IndiProperty &property /**< [in] Observed state update. */ ) override;

    /// Invalidate deleted capabilities, including device-wide and FSM deletions.
    void handleDelProperty( const pcf::IndiProperty &property /**< [in] Property or device deletion. */ ) override;

  public slots:
    /// Render acknowledged state and independent reconstruction/training availability.
    void updateGUI();

    /// Submit the staged catalog element; selecting a combo item alone sends nothing.
    void on_buttonEnableNN_clicked();

    /// Load the staged saved pair without starting learning or changing enable state.
    void on_buttonSetPolicy_clicked();

    /// Ask for an optional label and save the last acknowledged policy/dynamics pair.
    void on_buttonSavePolicy_clicked();

    /// Request the opposite of the observed reconstructor, with the expected source state.
    void on_buttonReconstructor_clicked();

    /// Request a shared modulation-link preference without moving the modulator.
    void on_checkModulation_clicked( bool checked /**< [in] Requested linking preference. */ );

    /// Request a whole-app Start/Stop through the native routing worker.
    void on_buttonRunPO4AO_clicked();

    /// Request PO4AO activation or coordinated disable according to observed control state.
    void on_buttonPO4AO_clicked();

    /// Reset the runtime policy and, when connected, the managed training session.
    void on_buttonReset_clicked();

    /// Request noisy warmup recollection under the current controller.
    void on_buttonWarmup_clicked();

    /// Request start/resume or cooperative pause according to observed learning state.
    void on_buttonLearning_clicked();

    /// Report missing acknowledgement without retrying or pretending the operation failed remotely.
    void acknowledgementTimeout();

  private:
    Ui::learningCtrl m_ui; ///< Qt Designer layout, using standard MagAO-X controls.

    /// Determine whether the optional trainer is live and controls this C++ device.
    bool trainerReady() const;

    /// Preserve unresolved operations when optional-trainer availability changes the command route.
    void updateRoute( bool managed /**< [in] Whether the optional trainer currently owns GUI commands. */ );

    /// Render native inference controls without depending on trainer state.
    void updateRuntimeGUI();

    /// Observe independent runtime/trainer progress; cached values do not renew freshness.
    void observeHeartbeat( const pcf::IndiProperty &property /**< [in] Newly received message. */ );

    /// Look up an observed text element, returning empty when it is unavailable.
    std::string value( const std::string &device, /**< [in] Source device. */
                       const std::string &name,   /**< [in] Property name. */
                       const std::string &element /**< [in] Element name. */ ) const;

    /// Convert machine state codes into compact operator-facing labels.
    static QString stateLabel( const std::string &state /**< [in] Observed state code, empty if unknown. */ );

    /// Look up an observed companion status field.
    std::string status( const std::string &element /**< [in] Status element name. */ ) const;

    /// Check that a defined writable command has the expected type and elements.
    bool
    commandAvailable( const std::string                  &name,     /**< [in] Command property. */
                      pcf::IndiProperty::Type             type,     /**< [in] Expected INDI type. */
                      std::initializer_list<const char *> elements, /**< [in] Required elements. */
                      const std::string &device = "",               /**< [in] Source device; empty selects trainer. */
                      bool supersede = false /**< [in] Permit native Reset to supersede control selection. */ ) const;

    /// Determine whether this domain has a pending or observed Busy command.
    bool busy( std::initializer_list<const char *> names, /**< [in] Commands sharing an operation domain. */
               const std::string &device = "" /**< [in] Source device; empty selects trainer. */ ) const;

    /// Determine whether the independent NN control endpoint is live.
    bool nnReady() const;

    /// Read the independent NN handover status.
    std::string nnStatus( const std::string &element /**< [in] Status element name. */ ) const;

    /// Synchronize catalog labels while preserving the user's unsent selection.
    void syncModels();

    /// Refresh the saved catalog while retaining the operator's staged identity.
    void syncPolicies();

    /// Render the optional trainer's numeric and saved-library controls.
    void updateTrainingControls( bool ready /**< [in] Whether the trainer has fresh readback. */ );

    /// Submit one scalar target using the standard status entry editing behavior.
    void submitParameter( const std::string &name, /**< [in] INDI numeric property. */
                          statusLineEdit    *entry /**< [in] Standard editable status field. */ );

    /// Observe the Busy then terminal command acknowledgement sequence.
    void acknowledge( const pcf::IndiProperty &property /**< [in] Device readback. */ );

    /// Latch a local request before sending it, preventing duplicate clicks until acknowledgement.
    void submit( const pcf::IndiProperty &property /**< [in] Explicit command payload. */ );

    /// Construct a single-element switch request for a defined device command.
    void submitSwitch( const std::string &name,    /**< [in] Command property. */
                       const std::string &element, /**< [in] Switch element. */
                       bool               on,      /**< [in] Explicit requested value. */
                       const std::string &device = "" /**< [in] Destination; empty selects trainer. */ );
};

inline learningCtrl::learningCtrl( const std::string &device,
                                   QWidget           *parent,
                                   const std::string &runtimeDevice,
                                   const std::string &nnDevice )
    : xWidget( parent ), m_device( device ), m_runtimeDevice( runtimeDevice ), m_nnDevice( nnDevice )
{
    m_ui.setupUi( this );
    setWindowTitle( "PO4AO / NN" );
    setXwFont( this );
    for( auto *label : { m_ui.labelCommandRegularization, m_ui.labelDeltaRegularization, m_ui.labelLearningRate } )
        label->setMinimumHeight( 2 * label->fontMetrics().height() );
    m_statusTimer.setInterval( 500 );
    connect( &m_statusTimer, &QTimer::timeout, this, &learningCtrl::updateGUI );
    m_statusTimer.start();
    m_ackTimer.setSingleShot( true );
    m_ackTimer.setInterval( 10000 );
    connect( &m_ackTimer, &QTimer::timeout, this, &learningCtrl::acknowledgementTimeout );
    connect(
        m_ui.comboEngine, QOverload<int>::of( &QComboBox::currentIndexChanged ), this, [this]( int ) { updateGUI(); } );
    connect( m_ui.comboPolicy,
             QOverload<int>::of( &QComboBox::currentIndexChanged ),
             this,
             [this]( int index )
             {
                 if( index >= 0 )
                     m_policySelection = m_ui.comboPolicy->currentData().toString();
                 updateGUI();
             } );
    for( const auto &entry :
         std::map<std::string, statusLineEdit *>{ { "command_regularization", m_ui.entryCommandRegularization },
                                                  { "delta_policy_regularization", m_ui.entryDeltaRegularization },
                                                  { "policy_learning_rate", m_ui.entryLearningRate } } )
        connect( entry.second,
                 &QLineEdit::returnPressed,
                 this,
                 [this, entry]() { submitParameter( entry.first, entry.second ); } );
    updateGUI();
}

inline void learningCtrl::subscribe()
{
    if( m_parent == nullptr )
        return;
    for( const auto *name : { "fsm",
                              "devices",
                              "po4ao_enable",
                              "reset",
                              "recollect",
                              "learning",
                              "status",
                              "progress",
                              "training_config",
                              "command_regularization",
                              "delta_policy_regularization",
                              "policy_learning_rate",
                              "policy_catalog",
                              "set_policy",
                              "save_policy" } )
        m_parent->addSubscriberProperty( this, m_device, name );
    for( const auto *name : { "fsm",
                              "run_enable",
                              "run_status",
                              "control_enable",
                              "reset_model",
                              "recollect_warmup",
                              "reload_engine",
                              "model_status",
                              "collection_status" } )
        m_parent->addSubscriberProperty( this, m_runtimeDevice, name );
    if( !m_nnDevice.empty() )
        for( const auto *name : { "fsm",
                                  "model",
                                  "model_status",
                                  "nn_activate",
                                  "reconstructor",
                                  "link_modulation",
                                  "reconstruction_status" } )
            m_parent->addSubscriberProperty( this, m_nnDevice, name );
}

inline void learningCtrl::onConnect()
{
    m_connected = true;
    updateGUI();
}

inline void learningCtrl::onDisconnect()
{
    m_connected = false;
    m_policySelection.clear();
    m_haveHeartbeat = false;
    m_heartbeatAge.invalidate();
    m_runtimeAge.invalidate();
    m_nnAge.invalidate();
    m_nnHeartbeat.clear();
    m_nnInstance.clear();
    m_runtimeInstance.clear();
    m_runtimeSequence.clear();
    m_properties.clear();
    m_defined.clear();
    m_pending.clear();
    m_routePending.clear();
    m_managed            = false;
    m_nativeResetPending = false;
    m_message.clear();
    m_ackTimer.stop();
    syncModels();
    syncPolicies();
    updateGUI();
}

inline std::string
learningCtrl::value( const std::string &device, const std::string &name, const std::string &element ) const
{
    const auto found = m_properties.find( device + "." + name );
    if( found == m_properties.end() || !found->second.find( element ) )
        return {};
    return found->second[element].get();
}

inline QString learningCtrl::stateLabel( const std::string &state )
{
    if( state.empty() )
        return "Unknown";
    if( state == "nn" )
        return "CNN";
    if( state == "cacao" )
        return "CACAO";
    if( state == "po4ao" )
        return "PO4AO";
    if( state == "waiting" )
        return "Waiting for data";
    auto label = QString::fromStdString( state );
    label[0]   = label[0].toUpper();
    return label;
}

inline std::string learningCtrl::status( const std::string &element ) const
{
    return value( m_device, "status", element );
}

inline bool learningCtrl::commandAvailable( const std::string                  &name,
                                            pcf::IndiProperty::Type             type,
                                            std::initializer_list<const char *> elements,
                                            const std::string                  &device,
                                            bool                                supersede ) const
{
    const auto key   = ( device.empty() ? m_device : device ) + "." + name;
    const auto found = m_properties.find( key );
    if( !m_connected || !m_defined.count( key ) || found == m_properties.end() || found->second.getType() != type ||
        found->second.getPerm() == pcf::IndiProperty::ReadOnly || !found->second.hasValidPerm() ||
        ( !supersede && ( found->second.getState() == pcf::IndiProperty::Busy || m_pending.count( key ) ) ) )
        return false;
    for( const auto *element : elements )
        if( !found->second.find( element ) )
            return false;
    return true;
}

inline bool learningCtrl::busy( std::initializer_list<const char *> names, const std::string &device ) const
{
    for( const auto *name : names )
    {
        const auto key   = ( device.empty() ? m_device : device ) + "." + name;
        const auto found = m_properties.find( key );
        if( m_pending.count( key ) ||
            ( found != m_properties.end() && found->second.getState() == pcf::IndiProperty::Busy ) )
            return true;
    }
    return false;
}

inline std::string learningCtrl::nnStatus( const std::string &element ) const
{
    return value( m_nnDevice, "reconstruction_status", element );
}

inline bool learningCtrl::nnReady() const
{
    const auto fsm = value( m_nnDevice, "fsm", "state" );
    return m_connected && m_nnAge.isValid() && m_nnAge.elapsed() < 5000 && !m_nnInstance.empty() &&
           m_defined.count( m_nnDevice + ".reconstruction_status" ) && ( fsm == "READY" || fsm == "OPERATING" );
}

inline void learningCtrl::syncModels()
{
    QSignalBlocker blocker( m_ui.comboEngine );
    const auto     selected = m_ui.comboEngine->currentData().toString();
    m_ui.comboEngine->clear();
    const auto key   = m_nnDevice + ".model";
    const auto found = m_properties.find( key );
    if( m_defined.count( key ) && found != m_properties.end() && found->second.getType() == pcf::IndiProperty::Switch )
    {
        for( const auto &entry : found->second.getElements() )
        {
            const auto label = entry.second.getLabel().empty() ? entry.first : entry.second.getLabel();
            m_ui.comboEngine->addItem( QString::fromStdString( label ), QString::fromStdString( entry.first ) );
            m_ui.comboEngine->setItemData(
                m_ui.comboEngine->count() - 1, QString::fromStdString( label ), Qt::ToolTipRole );
        }
    }
    const int previous = m_ui.comboEngine->findData( selected );
    m_ui.comboEngine->setCurrentIndex( previous ); // A redefinition never silently selects a different engine.
}

inline void learningCtrl::syncPolicies()
{
    QSignalBlocker blocker( m_ui.comboPolicy );
    const auto     selected = m_policySelection;
    m_ui.comboPolicy->clear();
    const auto key   = m_device + ".policy_catalog";
    const auto found = m_properties.find( key );
    if( m_defined.count( key ) && found != m_properties.end() && found->second.getType() == pcf::IndiProperty::Text )
        for( const auto &entry : found->second.getElements() )
        {
            if( entry.first == "none" )
                continue;
            const auto label = QString::fromStdString( entry.second.get() );
            m_ui.comboPolicy->addItem( label, QString::fromStdString( entry.first ) );
            m_ui.comboPolicy->setItemData( m_ui.comboPolicy->count() - 1, label, Qt::ToolTipRole );
        }
    m_ui.comboPolicy->setCurrentIndex( m_ui.comboPolicy->findData( selected ) );
}

inline void learningCtrl::updateTrainingControls( bool ready )
{
    const auto phase = status( "training_phase" );
    const auto runPhase = value(m_runtimeDevice,"run_status","phase");
    const bool routeQuiet = runPhase.empty() || runPhase == "unverified" || runPhase == "running" || runPhase == "stopped";
    const bool idle  = ready && routeQuiet && m_routePending.empty() && ( phase == "idle" || phase == "error" ) &&
                       !busy( { "po4ao_enable", "reset", "recollect", "learning", "set_policy" } );
    m_ui.comboPolicy->setEnabled( ready && m_ui.comboPolicy->count() > 0 );
    m_ui.comboPolicy->setToolTip( m_ui.comboPolicy->currentText() );
    m_ui.buttonSetPolicy->setEnabled( idle && m_ui.comboPolicy->currentIndex() >= 0 &&
                                      commandAvailable( "set_policy", pcf::IndiProperty::Text, { "id" } ) );
    m_ui.buttonSavePolicy->setEnabled( ready && status( "save_available" ) == "yes" &&
                                       commandAvailable( "save_policy", pcf::IndiProperty::Text, { "label" } ) );
    for( const auto &entry :
         std::map<std::string, statusLineEdit *>{ { "command_regularization", m_ui.entryCommandRegularization },
                                                  { "delta_policy_regularization", m_ui.entryDeltaRegularization },
                                                  { "policy_learning_rate", m_ui.entryLearningRate } } )
    {
        const bool perMode =
            entry.first == "delta_policy_regularization" && status( "delta_policy_mode" ) == "per_mode";
        const auto key   = m_device + "." + entry.first;
        const auto found = m_properties.find( key );
        const bool available =
            ready && commandAvailable( entry.first, pcf::IndiProperty::Number, { "current", "target" }, "", true );
        entry.second->setEnabled( available && !perMode );
        if( !available )
        {
            entry.second->QLineEdit::setText( "---" );
            continue;
        }
        if( perMode )
        {
            entry.second->QLineEdit::setText( "Per-mode: configured in file" );
            entry.second->setToolTip( "Per-mode: configured in file" );
            continue;
        }
        const auto current = value( m_device, entry.first, "current" );
        const auto target  = value( m_device, entry.first, "target" );
        const bool pending = found->second.getState() == pcf::IndiProperty::Busy || m_pending.count( key );
        const auto text    = QString::fromStdString( current );
        if( entry.second->currText() != text )
            entry.second->setTextChanged( text );
        entry.second->setToolTip( QString::fromStdString( "Current: " + current + "  Requested: " + target +
                                                          ( pending ? " (pending until iteration boundary)" : "" ) ) );
    }
}

inline void learningCtrl::submitParameter( const std::string &name, statusLineEdit *entry )
{
    if( !entry->isEnabled() || !trainerReady() )
        return;
    bool         valid  = false;
    const double number = entry->text().toDouble( &valid );
    if( !valid || !std::isfinite( number ) || number < 0 || ( name == "policy_learning_rate" && number == 0 ) )
    {
        m_message = "Enter a finite nonnegative penalty or a positive learning rate";
        updateGUI();
        return;
    }
    pcf::IndiProperty property( pcf::IndiProperty::Number, m_device, name );
    property.add( pcf::IndiElement( "target", number ) );
    entry->stopEditing();
    submit( property );
}

inline void learningCtrl::on_buttonSetPolicy_clicked()
{
    if( !m_ui.buttonSetPolicy->isEnabled() )
        return;
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_device, "set_policy" );
    property.add( pcf::IndiElement( "id", m_ui.comboPolicy->currentData().toString().toStdString() ) );
    submit( property );
}

inline void learningCtrl::on_buttonSavePolicy_clicked()
{
    if( !m_ui.buttonSavePolicy->isEnabled() )
        return;
    bool       accepted = false;
    const auto label =
        QInputDialog::getText( this, "Save policy", "Label (optional):", QLineEdit::Normal, "", &accepted );
    if( !accepted )
        return;
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_device, "save_policy" );
    property.add( pcf::IndiElement( "label", label.trimmed().toStdString() ) );
    submit( property );
}

inline void learningCtrl::acknowledge( const pcf::IndiProperty &property )
{
    const auto resetKey = m_runtimeDevice + ".reset_model";
    if( m_nativeResetPending && property.getDevice() == m_runtimeDevice )
    {
        bool       requestedValid = false, appliedValid = false;
        const auto requested = QString::fromStdString( value( m_runtimeDevice, "model_status", "reset_request" ) )
                                   .toULongLong( &requestedValid );
        const auto applied   = QString::fromStdString( value( m_runtimeDevice, "model_status", "reset_applied" ) )
                                   .toULongLong( &appliedValid );
        if( requestedValid && requested > m_nativeResetAfter )
        {
            m_pending[resetKey] = true; // Generation advancement acknowledges even a coalesced Busy transition.
            m_message.clear();
            const auto control   = value( m_runtimeDevice, "model_status", "control_request" );
            const auto selection = m_properties.find( m_runtimeDevice + ".control_enable" );
            if( appliedValid && applied == requested && !control.empty() &&
                control == value( m_runtimeDevice, "model_status", "control_applied" ) &&
                value( m_runtimeDevice, "model_status", "control_target" ) == "off" &&
                selection != m_properties.end() && selection->second.find( "toggle" ) &&
                selection->second["toggle"].getSwitchState() == pcf::IndiElement::Off &&
                selection->second.getState() == pcf::IndiProperty::Ok )
            {
                m_pending.erase( resetKey );
                m_nativeResetPending = false;
            }
        }
    }
    auto pending = m_pending.find( property.createUniqueKey() );
    if( pending != m_pending.end() )
    {
        if( property.getState() == pcf::IndiProperty::Busy )
        {
            pending->second = true;
            m_message.clear();
        }
        else if( property.getState() == pcf::IndiProperty::Alert ||
                 ( pending->second && property.getState() == pcf::IndiProperty::Ok &&
                   !( m_nativeResetPending && pending->first == resetKey ) ) )
        {
            m_message = property.getState() == pcf::IndiProperty::Alert ? "Request failed; check backend status" : "";
            if( pending->first == resetKey )
                m_nativeResetPending = false;
            m_pending.erase( pending );
        }
    }
    bool waiting = false;
    for( const auto &request : m_pending )
        waiting |= !request.second;
    if( !waiting )
        m_ackTimer.stop();
}

inline void learningCtrl::updateRoute( bool managed )
{
    if( managed != m_managed )
    {
        const auto &previous = m_managed ? m_device : m_runtimeDevice;
        for( const auto *name : { "po4ao_enable",
                                  "reset",
                                  "recollect",
                                  "learning",
                                  "set_policy",
                                  "control_enable",
                                  "reset_model",
                                  "recollect_warmup",
                                  "reload_engine" } )
            if( busy( { name }, previous ) )
                m_routePending.insert( previous + "." + name );
        const auto phase = status( "training_phase" );
        if( m_managed && !phase.empty() && phase != "idle" && phase != "error" )
            m_routePending.insert( m_device + ".status" ); // Automatic publication has no command vector.
        m_managed = managed;
    }
    const auto prefix = ( managed ? m_device : m_runtimeDevice ) + ".";
    for( auto pending = m_routePending.begin(); pending != m_routePending.end(); )
    {
        const auto property = m_properties.find( *pending );
        const auto phase    = status( "training_phase" );
        const bool phasePending =
            *pending == m_device + ".status" && !phase.empty() && phase != "idle" && phase != "error";
        if( pending->compare( 0, prefix.size(), prefix ) == 0 ||
            ( !m_pending.count( *pending ) && !phasePending &&
              ( property == m_properties.end() || property->second.getState() != pcf::IndiProperty::Busy ) ) )
            pending = m_routePending.erase( pending );
        else
            ++pending;
    }
}

inline bool learningCtrl::trainerReady() const
{
    const auto fsm = value( m_device, "fsm", "state" );
    return m_connected && m_heartbeatAge.isValid() && m_heartbeatAge.elapsed() < 5000 &&
           m_defined.count( m_device + ".status" ) && ( fsm == "READY" || fsm == "OPERATING" ) &&
           value( m_device, "devices", "po4ao" ) == m_runtimeDevice;
}

inline void learningCtrl::observeHeartbeat( const pcf::IndiProperty &property )
{
    if( property.getDevice() == m_nnDevice && property.getName() == "reconstruction_status" )
    {
        if( property.find( "instance" ) )
        {
            const auto instance = property["instance"].get();
            if( !m_nnInstance.empty() && m_nnInstance != instance )
            {
                pcf::IndiProperty request;
                request.setDevice( m_nnDevice );
                handleDelProperty( request );
                sendGetProperties( request );
            }
            m_nnInstance = instance;
        }
        if( property.find( "heartbeat" ) )
        {
            const auto heartbeat = property["heartbeat"].get();
            if( !m_nnHeartbeat.empty() && heartbeat != m_nnHeartbeat )
                m_nnAge.start();
            m_nnHeartbeat = heartbeat;
        }
    }

    if( property.getDevice() == m_runtimeDevice && property.getName() == "model_status" )
    {
        if( property.find( "instance" ) )
        {
            const auto instance = property["instance"].get();
            if( !m_runtimeInstance.empty() && instance != m_runtimeInstance )
            {
                pcf::IndiProperty request;
                request.setDevice( m_runtimeDevice );
                handleDelProperty( request );
                sendGetProperties( request );
            }
            m_runtimeInstance = instance;
        }
        if( property.find( "sequence" ) )
        {
            const auto sequence = property["sequence"].get();
            if( !m_runtimeSequence.empty() && sequence != m_runtimeSequence )
                m_runtimeAge.start();
            m_runtimeSequence = sequence;
        }
    }
    if( property.getDevice() != m_device || property.getName() != "status" || !property.find( "heartbeat" ) )
        return;
    const auto heartbeat = property["heartbeat"].get();
    if( m_haveHeartbeat && heartbeat != m_lastHeartbeat )
        m_heartbeatAge.start();
    m_lastHeartbeat = heartbeat;
    m_haveHeartbeat = true;
}

inline void learningCtrl::handleDefProperty( const pcf::IndiProperty &property )
{
    if( property.getDevice() != m_device && property.getDevice() != m_nnDevice &&
        property.getDevice() != m_runtimeDevice )
        return;
    observeHeartbeat( property );
    m_properties[property.createUniqueKey()] = property;
    m_defined.insert( property.createUniqueKey() );
    acknowledge( property );
    if( property.getDevice() == m_nnDevice && property.getName() == "model" )
        syncModels();
    syncPolicies();
    updateGUI();
}

inline void learningCtrl::handleSetProperty( const pcf::IndiProperty &property )
{
    if( property.getDevice() != m_device && property.getDevice() != m_nnDevice &&
        property.getDevice() != m_runtimeDevice )
        return;
    observeHeartbeat( property );
    const auto key   = property.createUniqueKey();
    auto       found = m_properties.find( key );
    if( found == m_properties.end() )
        m_properties.emplace( key, property );
    else
    {
        for( const auto &entry : property.getElements() )
        {
            if( found->second.find( entry.first ) )
                found->second[entry.first] = entry.second;
            else
                found->second.add( entry.second );
        }
        if( property.hasValidState() )
            found->second.setState( property.getState() );
    }
    acknowledge( property );
    if( property.getDevice() == m_device && property.getName() == "policy_catalog" )
        syncPolicies();
    updateGUI();
}

inline void learningCtrl::handleDelProperty( const pcf::IndiProperty &property )
{
    if( property.getDevice() != m_device && property.getDevice() != m_nnDevice &&
        property.getDevice() != m_runtimeDevice )
        return;
    const bool whole = property.getName().empty() || property.getName() == "fsm";
    for( auto entry = m_properties.begin(); entry != m_properties.end(); )
        if( entry->second.getDevice() == property.getDevice() &&
            ( whole || entry->second.getName() == property.getName() ) )
        {
            m_defined.erase( entry->first );
            entry = m_properties.erase( entry );
        }
        else
            ++entry;
    if( property.getDevice() == m_device )
    {
        if( whole || property.getName() == "status" )
        {
            m_haveHeartbeat = false;
            m_heartbeatAge.invalidate();
        }
    }
    if( property.getDevice() == m_nnDevice && ( whole || property.getName() == "reconstruction_status" ) )
    {
        m_nnAge.invalidate();
        m_nnHeartbeat.clear();
        m_nnInstance.clear();
    }
    if( property.getDevice() == m_runtimeDevice )
    {
        if( whole || property.getName() == "model_status" )
        {
            m_runtimeAge.invalidate();
            m_runtimeSequence.clear();
            m_runtimeInstance.clear();
            m_nativeResetPending = false;
            m_pending.erase( m_runtimeDevice + ".reset_model" );
        }
        if( property.getName() == "reset_model" )
            m_nativeResetPending = false;
    }
    const auto prefix = property.getDevice() + ".";
    for( auto pending = m_pending.begin(); pending != m_pending.end(); )
        if( ( whole && pending->first.compare( 0, prefix.size(), prefix ) == 0 ) ||
            pending->first == property.createUniqueKey() )
            pending = m_pending.erase( pending );
        else
            ++pending;
    if( m_pending.empty() )
        m_ackTimer.stop();
    syncModels();
    syncPolicies();
    updateGUI();
}

inline void learningCtrl::updateGUI()
{
    const bool fresh = m_heartbeatAge.isValid() && m_heartbeatAge.elapsed() < 5000;
    const bool ready = trainerReady();
    updateRoute( ready );
    m_ui.labelConnection->setText( !m_connected ? "Disconnected"
                                   : ready      ? "Ready"
                                   : !fresh     ? "Stale / waiting for heartbeat"
                                                : "Waiting for companion" );
    const bool nnLive              = nnReady();
    const auto source              = nnLive ? nnStatus( "reconstructor" ) : std::string( "unknown" );
    const bool knownSource         = source == "nn" || source == "cacao" || source == "stopped";
    const auto reconstructionPhase = nnStatus( "phase" );
    const bool reconstructionReady = nnLive && knownSource &&
                                     ( reconstructionPhase == "idle" || reconstructionPhase == "error" ) &&
                                     !busy( { "nn_activate", "reconstructor" }, m_nnDevice );
    const bool catalog             = m_defined.count( m_nnDevice + ".model" ) && m_ui.comboEngine->count() > 0;
    m_ui.comboEngine->setEnabled( nnLive && catalog && !busy( { "nn_activate", "reconstructor" }, m_nnDevice ) );
    m_ui.comboEngine->setToolTip( m_ui.comboEngine->currentText() );
    m_ui.buttonEnableNN->setEnabled(
        reconstructionReady && catalog && m_ui.comboEngine->currentIndex() >= 0 &&
        commandAvailable( "nn_activate", pcf::IndiProperty::Text, { "model" }, m_nnDevice ) );
    const bool loaded = !value( m_nnDevice, "model_status", "loaded" ).empty();
    m_ui.buttonReconstructor->setEnabled(
        reconstructionReady && ( source == "nn" || source == "stopped" || loaded ) &&
        commandAvailable( "reconstructor", pcf::IndiProperty::Text, { "target", "expected" }, m_nnDevice ) );
    m_ui.buttonReconstructor->setText( source == "cacao" ? "Switch to CNN" : "Switch to CACAO" );
    m_ui.labelReconstructor->setText( "Active: " + stateLabel( source ) );
    m_ui.labelReconstructor->setToolTip( QString::fromStdString(
        "Requested: " + value( m_nnDevice, "model_status", "requested" ) +
        "\nPrepared on host: " + value( m_nnDevice, "model_status", "loaded" ) +
        "\nGPU resident: " + value( m_nnDevice, "model_status", "resident" ) + "\nActive NN: " +
        value( m_nnDevice, "model_status", "active" ) + "\n" + nnStatus( "phase" ) + "\n" + nnStatus( "error" ) ) );
    {
        QSignalBlocker blocker( m_ui.checkModulation );
        const auto     link = m_properties.find( m_nnDevice + ".link_modulation" );
        m_ui.checkModulation->setChecked( link != m_properties.end() && link->second.find( "toggle" ) &&
                                          link->second["toggle"].getSwitchState() == pcf::IndiElement::On );
    }
    m_ui.checkModulation->setEnabled(
        nnLive && commandAvailable( "link_modulation", pcf::IndiProperty::Switch, { "toggle" }, m_nnDevice ) );
    m_ui.labelModulation->setText( "Modulation: " + QString::fromStdString( !nnLive || nnStatus( "modulation" ).empty()
                                                                                ? "Unknown"
                                                                                : nnStatus( "modulation" ) ) );

    const auto runPhase = value(m_runtimeDevice,"run_status","phase");
    const auto runProperty = m_properties.find(m_runtimeDevice+".run_enable");
    const bool appRunning = runProperty != m_properties.end() && runProperty->second.find("toggle") &&
                         runProperty->second["toggle"].getSwitchState() == pcf::IndiElement::On;
    m_ui.buttonRunPO4AO->setText(appRunning ? "Stop PO4AO" : "Start PO4AO");
    m_ui.buttonRunPO4AO->setEnabled(m_connected && m_runtimeAge.isValid() && m_runtimeAge.elapsed() < 5000 &&
        !m_runtimeInstance.empty() &&
        commandAvailable("run_enable",pcf::IndiProperty::Switch,{"toggle"},m_runtimeDevice) &&
        (runPhase == "unverified" || runPhase == "running" || runPhase == "stopped" || runPhase == "failed"));
    m_ui.labelRunPO4AO->setText(QString::fromStdString(runPhase + " | " + value(m_runtimeDevice,"run_status","route")));
    m_ui.labelRunPO4AO->setToolTip(QString::fromStdString(value(m_runtimeDevice,"run_status","error")));
    updateTrainingControls( ready );
    if( !ready )
    {
        updateRuntimeGUI();
        return;
    }

    const bool routeQuiet = runPhase.empty() || runPhase == "unverified" || runPhase == "running" || runPhase == "stopped";
    const bool runAvailable = runPhase.empty() || runPhase == "unverified" || runPhase == "running";
    const bool runtimePending  = !m_routePending.empty() || !routeQuiet;
    const auto controller      = ready ? status( "controller" ) : std::string( "unknown" );
    const bool enabled         = controller == "po4ao" || controller == "fallback";
    const bool knownController = enabled || controller == "integrator";
    const auto learning        = ready ? status( "learning" ) : std::string( "unknown" );
    const auto phase           = status( "training_phase" );
    const bool trainingReady   = ready && !runtimePending && ( phase == "idle" || phase == "error" ) &&
                                 !busy( { "po4ao_enable", "reset", "recollect", "learning", "set_policy" } );
    m_ui.buttonPO4AO->setText( enabled ? "Disable policy" : "Enable policy" );
    const bool disableReady = ready && !runtimePending && enabled && phase != "resetting" && phase != "reset_pending" &&
                              !busy( { "reset", "po4ao_enable" } );
    m_ui.buttonPO4AO->setEnabled( ( enabled ? disableReady : trainingReady && knownController ) &&
                                  commandAvailable( "po4ao_enable", pcf::IndiProperty::Switch, { "toggle" } ) );
    m_ui.labelController->setText( stateLabel( controller ) );
    m_ui.buttonReset->setEnabled( ready && !runtimePending && phase != "resetting" &&
                                  commandAvailable( "reset", pcf::IndiProperty::Switch, { "request" } ) );
    const auto collection = status( "collection" );
    m_ui.buttonWarmup->setEnabled( runAvailable && trainingReady && knownController && collection != "collecting" &&
                                   commandAvailable( "recollect", pcf::IndiProperty::Switch, { "request" } ) );
    QString collectionText = ready ? stateLabel( collection ) : "Unknown";
    if( ready && collection == "collecting" )
        collectionText +=
            QString::fromStdString( " " + status( "collection_frames" ) + "/" + status( "collection_target" ) );
    if( status( "warmup_required" ) == "yes" && collection != "collecting" )
        collectionText = "Fresh warmup required";
    m_ui.labelCollection->setText( collectionText );
    const bool running = learning == "running" || learning == "waiting";
    const bool paused  = learning == "paused" || learning == "completed";
    m_ui.buttonLearning->setText( running ? "Pause learning" : paused ? "Resume learning" : "Start learning" );
    m_ui.buttonLearning->setEnabled(
        runAvailable && trainingReady && collection != "collecting" && ( running || paused || learning == "idle" ) &&
        ( running || ( status( "warmup_required" ) != "yes" && status( "initialization_required" ) != "yes" ) ) &&
        commandAvailable( "learning", pcf::IndiProperty::Switch, { "run", "pause" } ) );
    m_ui.labelLearning->setText( status( "initialization_required" ) == "yes" ? "Set policy or Reset required"
                                                                              : stateLabel( learning ) );
    const auto active   = ready ? status( "active_policy" ) : std::string{};
    const auto retained = ready ? status( "retained_policy" ) : std::string{};
    m_ui.labelPolicy->setText(
        QString::fromStdString( "Policy: " + ( active.empty() ? "Unknown" : active ) +
                                ( retained.empty() || retained == active ? "" : "  Retained: " + retained ) ) );
    m_ui.labelPolicy->setToolTip( m_ui.labelPolicy->text() );
    const auto iteration = value( m_device, "progress", "iteration" );
    m_ui.labelIteration->setText( "Iteration: " + QString::fromStdString( iteration.empty() ? "—" : iteration ) );
    std::string message = m_message.empty() ? status( "message" ) : m_message;
    if( runtimePending )
        message = "Waiting for native PO4AO command completion";
    if( !status( "error" ).empty() )
        message = status( "error" );
    const auto configError   = value( m_device, "training_config", "error" );
    const auto pendingConfig = value( m_device, "training_config", "pending_reset" );
    if( !configError.empty() )
        message += "  Config: " + configError;
    else if( !pendingConfig.empty() )
        message += "  Configuration change requires Reset";
    if( busy( { "command_regularization", "delta_policy_regularization", "policy_learning_rate" } ) )
        message += paused ? "  Parameter edits pending until learning resumes" : "  Parameter edits pending";
    if( nnReady() && !nnStatus( "error" ).empty() )
        message += "  NN: " + nnStatus( "error" );
    m_ui.labelStatus->setText( QString::fromStdString( message.empty() ? "Status: —" : message ) );
    m_ui.labelStatus->setToolTip( QString::fromStdString( message ) );
}

inline void learningCtrl::updateRuntimeGUI()
{
    const auto fsm            = value( m_runtimeDevice, "fsm", "state" );
    const bool fresh          = m_runtimeAge.isValid() && m_runtimeAge.elapsed() < 5000;
    const auto runPhase = value(m_runtimeDevice,"run_status","phase");
    const bool routeQuiet = runPhase.empty() || runPhase == "unverified" || runPhase == "running" || runPhase == "stopped";
    const bool runAvailable = runPhase.empty() || runPhase == "unverified" || runPhase == "running";
    const bool trainerPending = !m_routePending.empty() || !routeQuiet;
    const bool ready = m_connected && fresh && !m_runtimeInstance.empty() &&
                       m_defined.count( m_runtimeDevice + ".model_status" ) && ( fsm == "READY" || fsm == "OPERATING" );
    m_ui.labelConnection->setText( !m_connected ? "Disconnected"
                                   : ready      ? "PO4AO ready / trainer unavailable"
                                   : nnReady()  ? "NN ready / PO4AO unavailable"
                                   : !fresh     ? "Stale / waiting for heartbeat"
                                                : "Waiting for PO4AO" );
    const auto selection = m_properties.find( m_runtimeDevice + ".control_enable" );
    const bool known     = selection != m_properties.end() && selection->second.find( "toggle" ) &&
                           ( selection->second["toggle"].getSwitchState() == pcf::IndiElement::On ||
                             selection->second["toggle"].getSwitchState() == pcf::IndiElement::Off );
    const bool enabled   = known && selection->second["toggle"].getSwitchState() == pcf::IndiElement::On;
    m_ui.buttonPO4AO->setText( enabled ? "Disable policy" : "Enable policy" );
    m_ui.buttonPO4AO->setEnabled(
        ready && !trainerPending && known && !busy( { "reset_model" }, m_runtimeDevice ) &&
        commandAvailable( "control_enable", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice ) );
    bool resetKnown = false;
    QString::fromStdString( value( m_runtimeDevice, "model_status", "reset_request" ) ).toULongLong( &resetKnown );
    const auto target = value( m_runtimeDevice, "model_status", "control_target" );
    m_ui.buttonReset->setEnabled(
        ready && !trainerPending && resetKnown && ( target == "on" || target == "off" ) &&
        commandAvailable( "control_enable", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice, true ) &&
        commandAvailable( "reset_model", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice ) );
    m_ui.buttonWarmup->setEnabled(
        runAvailable && ready && !trainerPending && !busy( { "control_enable", "reset_model" }, m_runtimeDevice ) &&
        commandAvailable( "recollect_warmup", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice ) );
    m_ui.buttonLearning->setEnabled( false );
    m_ui.buttonLearning->setText( "Start learning" );
    m_ui.labelLearning->setText( "Trainer unavailable" );
    m_ui.labelIteration->setText( "Iteration: —" );
    const auto controller = ready ? value( m_runtimeDevice, "model_status", "effective_controller" ) : std::string{};
    m_ui.labelController->setText( stateLabel( controller ) );
    QString collection = ready ? stateLabel( value( m_runtimeDevice, "collection_status", "state" ) ) : "Unknown";
    if( ready )
        collection += QString::fromStdString( " " + value( m_runtimeDevice, "collection_status", "frames" ) + "/" +
                                              value( m_runtimeDevice, "collection_status", "target_frames" ) );
    m_ui.labelCollection->setText( collection );
    const auto loaded    = ready ? value( m_runtimeDevice, "model_status", "active_id" ) : std::string{};
    const auto effective = ready ? value( m_runtimeDevice, "model_status", "effective_id" ) : std::string{};
    m_ui.labelPolicy->setText( QString::fromStdString( "Loaded: " + ( loaded.empty() ? "Unknown" : loaded ) +
                                                       "  Active: " + ( effective.empty() ? "Unknown" : effective ) ) );
    m_ui.labelPolicy->setToolTip( m_ui.labelPolicy->text() );
    std::string message = trainerPending ? "Trainer request unresolved; awaiting acknowledgement" : m_message;
    const auto  error   = ready ? value( m_runtimeDevice, "model_status", "reload_error" ) : std::string{};
    if( !error.empty() )
        message = error;
    else if( message.empty() )
        message = m_nativeResetPending ? "Reset pending: waiting for runtime completion"
                  : ready              ? "Reload: " + value( m_runtimeDevice, "model_status", "reload" )
                                       : "Waiting for runtime readback";
    if( nnReady() && !nnStatus( "error" ).empty() )
        message += "  NN: " + nnStatus( "error" );
    m_ui.labelStatus->setText( QString::fromStdString( message ) );
    m_ui.labelStatus->setToolTip( m_ui.labelStatus->text() );
}

inline void learningCtrl::submit( const pcf::IndiProperty &property )
{
    m_pending[property.createUniqueKey()] = false;
    m_message                             = "Awaiting acknowledgement";
    if( !m_ackTimer.isActive() )
        m_ackTimer.start();
    updateGUI();
    sendNewProperty( property );
}

inline void
learningCtrl::submitSwitch( const std::string &name, const std::string &element, bool on, const std::string &device )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch, device.empty() ? m_device : device, name );
    property.add( pcf::IndiElement( element, on ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    submit( property );
}

inline void learningCtrl::on_buttonEnableNN_clicked()
{
    updateGUI();
    if( !m_ui.buttonEnableNN->isEnabled() )
        return;
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_nnDevice, "nn_activate" );
    property.add( pcf::IndiElement( "model", m_ui.comboEngine->currentData().toString().toStdString() ) );
    submit( property );
}

inline void learningCtrl::on_buttonReconstructor_clicked()
{
    updateGUI();
    if( !m_ui.buttonReconstructor->isEnabled() )
        return;
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_nnDevice, "reconstructor" );
    property.add( pcf::IndiElement( "target", nnStatus( "reconstructor" ) == "cacao" ? "nn" : "cacao" ) );
    property.add( pcf::IndiElement( "expected", nnStatus( "reconstructor" ) ) );
    submit( property );
}

inline void learningCtrl::on_checkModulation_clicked( bool checked )
{
    if( m_ui.checkModulation->isEnabled() )
        submitSwitch( "link_modulation", "toggle", checked, m_nnDevice );
}

inline void learningCtrl::on_buttonPO4AO_clicked()
{
    updateGUI();
    if( !m_ui.buttonPO4AO->isEnabled() )
        return;
    if( m_managed )
        submitSwitch( "po4ao_enable", "toggle", status( "controller" ) == "integrator" );
    else
        submitSwitch( "control_enable",
                      "toggle",
                      m_properties.at( m_runtimeDevice + ".control_enable" )["toggle"].getSwitchState() !=
                          pcf::IndiElement::On,
                      m_runtimeDevice );
}

inline void learningCtrl::on_buttonRunPO4AO_clicked()
{
    updateGUI();
    if(!m_ui.buttonRunPO4AO->isEnabled()) return;
    const bool running = m_properties.at(m_runtimeDevice+".run_enable")["toggle"].getSwitchState() == pcf::IndiElement::On;
    submitSwitch("run_enable","toggle",!running,m_runtimeDevice);
}

inline void learningCtrl::on_buttonReset_clicked()
{
    updateGUI();
    if( !m_ui.buttonReset->isEnabled() )
        return;
    if( m_managed )
        submitSwitch( "reset", "request", true );
    else
    {
        m_nativeResetAfter =
            QString::fromStdString( value( m_runtimeDevice, "model_status", "reset_request" ) ).toULongLong();
        m_nativeResetPending = true;
        m_ackTimer.start();
        m_pending.erase( m_runtimeDevice + ".control_enable" ); // Reset owns completion of the superseded selection.
        pcf::IndiProperty control( pcf::IndiProperty::Switch, m_runtimeDevice, "control_enable" );
        control.add( pcf::IndiElement( "toggle", pcf::IndiElement::Off ) );
        sendNewProperty( control ); // Off can be idempotent; its own Busy edge is not required.
        submitSwitch( "reset_model", "toggle", true, m_runtimeDevice );
    }
}

inline void learningCtrl::on_buttonWarmup_clicked()
{
    updateGUI();
    if( !m_ui.buttonWarmup->isEnabled() )
        return;
    if( m_managed )
        submitSwitch( "recollect", "request", true );
    else
        submitSwitch( "recollect_warmup", "toggle", true, m_runtimeDevice );
}

inline void learningCtrl::on_buttonLearning_clicked()
{
    updateGUI();
    if( !m_ui.buttonLearning->isEnabled() )
        return;
    const bool        run = status( "learning" ) != "running" && status( "learning" ) != "waiting";
    pcf::IndiProperty property( pcf::IndiProperty::Switch, m_device, "learning" );
    property.add( pcf::IndiElement( "run", run ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    property.add( pcf::IndiElement( "pause", run ? pcf::IndiElement::Off : pcf::IndiElement::On ) );
    submit( property );
}

inline void learningCtrl::acknowledgementTimeout()
{
    m_message = "No acknowledgement; operation state is unknown. Check device status.";
    updateGUI(); // Keep the request latched. A timeout is not permission to resend it.
}

} // namespace xqt
#include "moc_learningCtrl.cpp"
#endif
