/** \file learningCtrl.hpp
 * \brief PO4AO inference and optional managed training through the shared INDI publisher.
 */
#ifndef learningCtrl_hpp
#define learningCtrl_hpp

#include <map>
#include <set>
#include <string>

#include <QElapsedTimer>
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

    std::string m_nnDevice; ///< NN device discovered through the companion's devices property.

    bool m_connected{ false }; ///< Transport connection; readiness also needs property definitions.

    std::map<std::string, pcf::IndiProperty>
        m_properties; ///< Definitions and merged partial updates, owned by Qt thread.

    std::set<std::string> m_defined; ///< Properties with observed definitions, including command permissions.

    std::map<std::string, bool>
        m_pending; ///< Requests keyed by device.property; value records an observed Busy acknowledgement.

    std::string m_message; ///< Local request/acknowledgement error, separate from backend status.

    QTimer m_statusTimer; ///< Re-evaluates expiry even when no INDI message arrives.

    QElapsedTimer m_heartbeatAge; ///< Monotonic age since an observed advancing companion heartbeat.

    std::string m_lastHeartbeat; ///< Last sequence, used to reject unchanged cached definitions.

    bool m_haveHeartbeat{ false }; ///< A baseline sequence has been observed in this device lifetime.

    QTimer m_ackTimer; ///< Bounds the wait for initial acknowledgement; never retries a command.

  public:
    /// Construct one panel for C++ inference and an optional companion.
    explicit learningCtrl( const std::string &device,                  /**< [in] Companion device name. */
                           QWidget           *parent        = nullptr, /**< [in] Optional Qt parent. */
                           const std::string &runtimeDevice = "po4ao" /**< [in] C++ runtime device. */ );

    /// Attach named property subscriptions through the existing publisher.
    void subscribe() override;

    /// Mark transport connection without inferring device readiness.
    void onConnect() override;

    /// Invalidate cached states and pending requests on disconnect.
    void onDisconnect() override;

    /// Replace a definition and discover a changed NN device mapping.
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

    /// Request the opposite of the observed reconstructor, with the expected source state.
    void on_buttonReconstructor_clicked();

    /// Request a shared modulation-link preference without moving the modulator.
    void on_checkModulation_clicked( bool checked /**< [in] Requested linking preference. */ );

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
    bool commandAvailable( const std::string                  &name,     /**< [in] Command property. */
                           pcf::IndiProperty::Type             type,     /**< [in] Expected INDI type. */
                           std::initializer_list<const char *> elements, /**< [in] Required elements. */
                           const std::string &device = "" /**< [in] Source device; empty selects trainer. */ ) const;

    /// Determine whether this domain has a pending or observed Busy command.
    bool busy( std::initializer_list<const char *> names, /**< [in] Commands sharing an operation domain. */
               const std::string &device = "" /**< [in] Source device; empty selects trainer. */ ) const;

    /// Update dynamic NN subscriptions after the companion's device mapping changes.
    void syncNNDevice();

    /// Synchronize catalog labels while preserving the user's unsent selection.
    void syncModels();

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

inline learningCtrl::learningCtrl( const std::string &device, QWidget *parent, const std::string &runtimeDevice )
    : xWidget( parent ), m_device( device ), m_runtimeDevice( runtimeDevice )
{
    m_ui.setupUi( this );
    setWindowTitle( "PO4AO / NN" );
    setXwFont( this );
    m_statusTimer.setInterval( 500 );
    connect( &m_statusTimer, &QTimer::timeout, this, &learningCtrl::updateGUI );
    m_statusTimer.start();
    m_ackTimer.setSingleShot( true );
    m_ackTimer.setInterval( 10000 );
    connect( &m_ackTimer, &QTimer::timeout, this, &learningCtrl::acknowledgementTimeout );
    connect(
        m_ui.comboEngine, QOverload<int>::of( &QComboBox::currentIndexChanged ), this, [this]( int ) { updateGUI(); } );
    updateGUI();
}

inline void learningCtrl::subscribe()
{
    if( m_parent == nullptr )
        return;
    for( const auto *name : { "fsm",
                              "devices",
                              "nn_activate",
                              "reconstructor",
                              "link_modulation",
                              "po4ao_enable",
                              "reset",
                              "recollect",
                              "learning",
                              "status",
                              "progress",
                              "training_config" } )
        m_parent->addSubscriberProperty( this, m_device, name );
    for( const auto *name :
         { "fsm", "control_enable", "reset_model", "recollect_warmup", "model_status", "collection_status" } )
        m_parent->addSubscriberProperty( this, m_runtimeDevice, name );
    if( !m_nnDevice.empty() )
        for( const auto *name : { "fsm", "model", "model_status" } )
            m_parent->addSubscriberProperty( this, m_nnDevice, name );
}

inline void learningCtrl::onConnect()
{
    m_connected = true;
    updateGUI();
}

inline void learningCtrl::onDisconnect()
{
    m_connected     = false;
    m_haveHeartbeat = false;
    m_heartbeatAge.invalidate();
    m_runtimeAge.invalidate();
    m_runtimeInstance.clear();
    m_runtimeSequence.clear();
    m_properties.clear();
    m_defined.clear();
    m_pending.clear();
    m_message.clear();
    m_ackTimer.stop();
    syncModels();
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
        return "NN";
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
                                            const std::string                  &device ) const
{
    const auto key   = ( device.empty() ? m_device : device ) + "." + name;
    const auto found = m_properties.find( key );
    if( !m_connected || !m_defined.count( key ) || found == m_properties.end() || found->second.getType() != type ||
        found->second.getPerm() == pcf::IndiProperty::ReadOnly || !found->second.hasValidPerm() ||
        found->second.getState() == pcf::IndiProperty::Busy || m_pending.count( key ) )
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

inline void learningCtrl::syncNNDevice()
{
    const auto name = value( m_device, "devices", "nn" );
    if( name == m_nnDevice )
        return;
    const auto old = m_nnDevice;
    m_nnDevice     = name;
    if( !old.empty() )
        for( const auto *prop : { "fsm", "model", "model_status" } )
        {
            if( m_parent )
                m_parent->unsubscribe( this, old, prop );
            m_properties.erase( old + "." + prop );
            m_defined.erase( old + "." + prop );
        }
    if( m_parent && !m_nnDevice.empty() )
        for( const auto *prop : { "fsm", "model", "model_status" } )
            m_parent->addSubscriberProperty( this, m_nnDevice, prop );
    syncModels();
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

inline void learningCtrl::acknowledge( const pcf::IndiProperty &property )
{
    auto pending = m_pending.find( property.createUniqueKey() );
    if( pending == m_pending.end() )
        return;
    if( property.getState() == pcf::IndiProperty::Busy )
    {
        pending->second = true;
        m_message.clear();
    }
    else if( property.getState() == pcf::IndiProperty::Alert ||
             ( pending->second && property.getState() == pcf::IndiProperty::Ok ) )
    {
        m_message = property.getState() == pcf::IndiProperty::Alert ? "Request failed; check backend status" : "";
        m_pending.erase( pending );
    }
    bool waiting = false;
    for( const auto &request : m_pending )
        waiting |= !request.second;
    if( !waiting )
        m_ackTimer.stop();
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
    if( property.getDevice() == m_device && property.getName() == "devices" )
        syncNNDevice();
    if( property.getDevice() == m_nnDevice && property.getName() == "model" )
        syncModels();
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
    if( property.getDevice() == m_device && property.getName() == "devices" )
        syncNNDevice();
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
        if( whole || property.getName() == "devices" )
            syncNNDevice();
    }
    if( property.getDevice() == m_runtimeDevice )
    {
        if( whole || property.getName() == "model_status" )
        {
            m_runtimeAge.invalidate();
            m_runtimeSequence.clear();
            m_runtimeInstance.clear();
        }
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
    updateGUI();
}

inline void learningCtrl::updateGUI()
{
    const bool fresh = m_heartbeatAge.isValid() && m_heartbeatAge.elapsed() < 5000;
    const bool ready = trainerReady();
    m_ui.labelConnection->setText( !m_connected ? "Disconnected"
                                   : ready      ? "Ready"
                                   : !fresh     ? "Stale / waiting for heartbeat"
                                                : "Waiting for companion" );
    const auto source              = ready ? status( "reconstructor" ) : std::string( "unknown" );
    const bool knownSource         = source == "nn" || source == "cacao" || source == "stopped";
    const auto reconstructionPhase = status( "reconstruction_phase" );
    const bool reconstructionReady = ready && knownSource &&
                                     ( reconstructionPhase == "idle" || reconstructionPhase == "error" ) &&
                                     !busy( { "nn_activate", "reconstructor" } );
    const bool catalog             = m_defined.count( m_nnDevice + ".model" ) && m_ui.comboEngine->count() > 0;
    m_ui.comboEngine->setEnabled( ready && catalog && !busy( { "nn_activate", "reconstructor" } ) );
    m_ui.comboEngine->setToolTip( m_ui.comboEngine->currentText() );
    m_ui.buttonEnableNN->setEnabled( reconstructionReady && catalog && m_ui.comboEngine->currentIndex() >= 0 &&
                                     commandAvailable( "nn_activate", pcf::IndiProperty::Text, { "model" } ) );
    const bool loaded = !value( m_nnDevice, "model_status", "loaded" ).empty();
    m_ui.buttonReconstructor->setEnabled(
        reconstructionReady && ( source == "nn" || source == "stopped" || loaded ) &&
        commandAvailable( "reconstructor", pcf::IndiProperty::Text, { "target", "expected" } ) );
    m_ui.buttonReconstructor->setText( source == "stopped" ? "Use CACAO" : "Switch reconstructor" );
    m_ui.labelReconstructor->setText( "Active: " + stateLabel( source ) );
    m_ui.labelReconstructor->setToolTip(
        QString::fromStdString( "Requested: " + value( m_nnDevice, "model_status", "requested" ) +
                                "\nLoaded: " + value( m_nnDevice, "model_status", "loaded" ) +
                                "\nActive NN: " + value( m_nnDevice, "model_status", "active" ) ) );
    {
        QSignalBlocker blocker( m_ui.checkModulation );
        const auto     link = m_properties.find( m_device + ".link_modulation" );
        m_ui.checkModulation->setChecked( link != m_properties.end() && link->second.find( "toggle" ) &&
                                          link->second["toggle"].getSwitchState() == pcf::IndiElement::On );
    }
    m_ui.checkModulation->setEnabled( ready &&
                                      commandAvailable( "link_modulation", pcf::IndiProperty::Switch, { "toggle" } ) );
    m_ui.labelModulation->setText( "Modulation: " + QString::fromStdString( !ready || status( "modulation" ).empty()
                                                                                ? "Unknown"
                                                                                : status( "modulation" ) ) );

    if( !ready )
    {
        updateRuntimeGUI();
        return;
    }

    const bool runtimePending  = m_pending.count( m_runtimeDevice + ".control_enable" ) ||
                                 m_pending.count( m_runtimeDevice + ".reset_model" ) ||
                                 m_pending.count( m_runtimeDevice + ".recollect_warmup" );
    const auto controller      = ready ? status( "controller" ) : std::string( "unknown" );
    const bool enabled         = controller == "po4ao" || controller == "fallback";
    const bool knownController = enabled || controller == "integrator";
    const auto learning        = ready ? status( "learning" ) : std::string( "unknown" );
    const auto phase           = status( "training_phase" );
    const bool trainingReady   = ready && !runtimePending && ( phase == "idle" || phase == "error" ) &&
                                 !busy( { "po4ao_enable", "reset", "recollect", "learning" } );
    m_ui.buttonPO4AO->setText( enabled ? "Disable PO4AO" : "Enable PO4AO" );
    const bool disableReady = ready && !runtimePending && enabled && phase != "resetting" && phase != "reset_pending" &&
                              !busy( { "reset", "po4ao_enable" } );
    m_ui.buttonPO4AO->setEnabled( ( enabled ? disableReady : trainingReady && knownController ) &&
                                  commandAvailable( "po4ao_enable", pcf::IndiProperty::Switch, { "toggle" } ) );
    m_ui.labelController->setText( stateLabel( controller ) );
    m_ui.buttonReset->setEnabled( ready && !runtimePending && phase != "resetting" &&
                                  commandAvailable( "reset", pcf::IndiProperty::Switch, { "request" } ) );
    const auto collection = status( "collection" );
    m_ui.buttonWarmup->setEnabled( trainingReady && knownController && collection != "collecting" &&
                                   commandAvailable( "recollect", pcf::IndiProperty::Switch, { "request" } ) );
    QString collectionText = ready ? stateLabel( collection ) : "Unknown";
    if( ready && collection == "collecting" )
        collectionText +=
            QString::fromStdString( " " + status( "collection_frames" ) + "/" + status( "collection_target" ) );
    m_ui.labelCollection->setText( collectionText );
    const bool running = learning == "running" || learning == "waiting";
    const bool paused  = learning == "paused" || learning == "completed";
    m_ui.buttonLearning->setText( running ? "Pause learning" : paused ? "Resume learning" : "Start learning" );
    m_ui.buttonLearning->setEnabled( trainingReady && collection != "collecting" &&
                                     ( running || paused || learning == "idle" ) &&
                                     commandAvailable( "learning", pcf::IndiProperty::Switch, { "run", "pause" } ) );
    m_ui.labelLearning->setText( stateLabel( learning ) );
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
    m_ui.labelStatus->setText( QString::fromStdString( message.empty() ? "Status: —" : message ) );
    m_ui.labelStatus->setToolTip( QString::fromStdString( message ) );
}

inline void learningCtrl::updateRuntimeGUI()
{
    const auto fsm            = value( m_runtimeDevice, "fsm", "state" );
    const bool fresh          = m_runtimeAge.isValid() && m_runtimeAge.elapsed() < 5000;
    const bool trainerPending = m_pending.count( m_device + ".po4ao_enable" ) ||
                                m_pending.count( m_device + ".reset" ) || m_pending.count( m_device + ".recollect" ) ||
                                m_pending.count( m_device + ".learning" );
    const bool ready = m_connected && fresh && !trainerPending && !m_runtimeInstance.empty() &&
                       m_defined.count( m_runtimeDevice + ".model_status" ) && ( fsm == "READY" || fsm == "OPERATING" );
    m_ui.labelConnection->setText( !m_connected ? "Disconnected"
                                   : ready      ? "PO4AO ready / trainer unavailable"
                                   : !fresh     ? "Stale / waiting for heartbeat"
                                                : "Waiting for PO4AO" );
    const auto selection = m_properties.find( m_runtimeDevice + ".control_enable" );
    const bool known     = selection != m_properties.end() && selection->second.find( "toggle" ) &&
                           ( selection->second["toggle"].getSwitchState() == pcf::IndiElement::On ||
                             selection->second["toggle"].getSwitchState() == pcf::IndiElement::Off );
    const bool enabled   = known && selection->second["toggle"].getSwitchState() == pcf::IndiElement::On;
    m_ui.buttonPO4AO->setText( enabled ? "Disable PO4AO" : "Enable PO4AO" );
    m_ui.buttonPO4AO->setEnabled(
        ready && known && !busy( { "reset_model" }, m_runtimeDevice ) &&
        commandAvailable( "control_enable", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice ) );
    m_ui.buttonReset->setEnabled(
        ready && known && !busy( { "control_enable" }, m_runtimeDevice ) &&
        ( !enabled ||
          commandAvailable( "control_enable", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice ) ) &&
        commandAvailable( "reset_model", pcf::IndiProperty::Switch, { "toggle" }, m_runtimeDevice ) );
    m_ui.buttonWarmup->setEnabled(
        ready && !busy( { "control_enable", "reset_model" }, m_runtimeDevice ) &&
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
        message =
            ready ? "Reload: " + value( m_runtimeDevice, "model_status", "reload" ) : "Waiting for runtime readback";
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
    if( !m_ui.buttonEnableNN->isEnabled() )
        return;
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_device, "nn_activate" );
    property.add( pcf::IndiElement( "model", m_ui.comboEngine->currentData().toString().toStdString() ) );
    submit( property );
}

inline void learningCtrl::on_buttonReconstructor_clicked()
{
    if( !m_ui.buttonReconstructor->isEnabled() )
        return;
    pcf::IndiProperty property( pcf::IndiProperty::Text, m_device, "reconstructor" );
    property.add( pcf::IndiElement( "target", status( "reconstructor" ) == "cacao" ? "nn" : "cacao" ) );
    property.add( pcf::IndiElement( "expected", status( "reconstructor" ) ) );
    submit( property );
}

inline void learningCtrl::on_checkModulation_clicked( bool checked )
{
    if( m_ui.checkModulation->isEnabled() )
        submitSwitch( "link_modulation", "toggle", checked );
}

inline void learningCtrl::on_buttonPO4AO_clicked()
{
    updateGUI();
    if( !m_ui.buttonPO4AO->isEnabled() )
        return;
    if( trainerReady() )
        submitSwitch( "po4ao_enable", "toggle", status( "controller" ) == "integrator" );
    else
        submitSwitch( "control_enable",
                      "toggle",
                      m_properties.at( m_runtimeDevice + ".control_enable" )["toggle"].getSwitchState() !=
                          pcf::IndiElement::On,
                      m_runtimeDevice );
}

inline void learningCtrl::on_buttonReset_clicked()
{
    updateGUI();
    if( !m_ui.buttonReset->isEnabled() )
        return;
    if( trainerReady() )
        submitSwitch( "reset", "request", true );
    else
    {
        if( m_properties.at( m_runtimeDevice + ".control_enable" )["toggle"].getSwitchState() == pcf::IndiElement::On )
            submitSwitch( "control_enable", "toggle", false, m_runtimeDevice );
        submitSwitch( "reset_model", "toggle", true, m_runtimeDevice );
    }
}

inline void learningCtrl::on_buttonWarmup_clicked()
{
    updateGUI();
    if( !m_ui.buttonWarmup->isEnabled() )
        return;
    if( trainerReady() )
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
