/** \file learningCtrl_test.cpp
 * \brief Offline shared-GUI commands, acknowledgements, lifecycle and layout tests.
 * \ingroup learningCtrlGUI_unit_test
 */
#define CATCH_CONFIG_RUNNER
#include "../../../../tests/testXWC.hpp"

#include <QApplication>
#include <QFile>
#include <QTextStream>
#include <QThread>

#include "../../../lib/tests/offlineIndiPublisher.hpp"
#include "../learningCtrl.hpp"

namespace libXWCTest
{
namespace learningCtrlGUITest
{
/// \cond TEST_HARNESS
using Publisher = guiTest::OfflinePublisher;

/// Deliver queued production publisher callbacks without sockets or sleeps.
void drain();

/// Create a text property with explicit permission and readback state.
pcf::IndiProperty
text( const std::string                                         &device, /**< [in] INDI device. */
      const std::string                                         &name,   /**< [in] Property name. */
      std::initializer_list<std::pair<std::string, std::string>> elements /**< [in] Text elements. */ );

/// Construct a native runtime switch readback with an explicit acknowledgement state.
pcf::IndiProperty
runtimeSwitch( const std::string                   &name,    /**< [in] Native command name. */
               bool                                 enabled, /**< [in] Applied toggle value. */
               pcf::IndiProperty::PropertyStateType state = pcf::IndiProperty::Ok /**< [in] Command state. */ );

/// Own an actual widget and the upstream offline publisher.
struct Fixture
{
    Publisher m_publisher; ///< Real dispatcher using a socket-free transport.

    xqt::learningCtrl m_widget{ "trainer", nullptr, "po4ao", "nn_test" }; ///< Production widget under test.

    /// Connect and define representative independent control domains.
    explicit Fixture( bool companion = true /**< [in] Define the optional trainer on startup. */ );

    /// Define native runtime state independently of any trainer.
    void runtime( bool               enabled  = false, /**< [in] Applied control selection. */
                  const std::string &instance = "runtime-1" /**< [in] Process identity. */ );

    /// Define independent NN capabilities without a trainer or PO4AO device.
    void nn();

    /// Redefine the representative device properties for all attached clients.
    void ready();

    /// Define the optional numeric controls and immutable saved policy catalog.
    void trainingControls();

    /// Define a property through the production queued dispatcher.
    void define( const pcf::IndiProperty &property /**< [in] Definition. */ );

    /// Send a sparse observed status update.
    void
    status( std::initializer_list<std::pair<std::string, std::string>> elements /**< [in] Changed status fields. */ );

    /// Send a coherent sparse native runtime status update.
    void runtimeStatus(
        std::initializer_list<std::pair<std::string, std::string>> elements /**< [in] Changed native fields. */ );

    /// Publish a command's observed property state without changing its definition.
    void acknowledge( const std::string                   &name, /**< [in] Command property. */
                      pcf::IndiProperty::PropertyStateType state /**< [in] Acknowledged state. */ );

    /// Delete a property or a whole device.
    void erase( const std::string &device, /**< [in] Device. */
                const std::string &name /**< [in] Property; empty deletes the device. */ );

    /// Return a named actual child control, asserting its Designer identity.
    template <typename T>
    T *control( const char *name /**< [in] Qt object name. */ );

    /// Complete a command after its mandatory Busy acknowledgement.
    void complete( const std::string &name /**< [in] Command property. */ );
};

void drain()
{
    QCoreApplication::processEvents();
}

pcf::IndiProperty text( const std::string                                         &device,
                        const std::string                                         &name,
                        std::initializer_list<std::pair<std::string, std::string>> elements )
{
    pcf::IndiProperty property( pcf::IndiProperty::Text, device, name );
    property.setPerm( pcf::IndiProperty::ReadWrite );
    property.setState( pcf::IndiProperty::Ok );
    for( const auto &entry : elements )
        property.add( pcf::IndiElement( entry.first, entry.second ) );
    return property;
}

pcf::IndiProperty runtimeSwitch( const std::string &name, bool enabled, pcf::IndiProperty::PropertyStateType state )
{
    pcf::IndiProperty property( pcf::IndiProperty::Switch, "po4ao", name );
    property.setPerm( pcf::IndiProperty::ReadWrite );
    property.setState( state );
    property.add( pcf::IndiElement( "toggle", enabled ? pcf::IndiElement::On : pcf::IndiElement::Off ) );
    return property;
}

void Fixture::runtime( bool enabled, const std::string &instance )
{
    define( text( "po4ao",
                  "model_status",
                  { { "instance", instance },
                    { "sequence", "0" },
                    { "control_target", enabled ? "on" : "off" },
                    { "control_request", "1" },
                    { "control_applied", "1" },
                    { "reset_request", "1" },
                    { "reset_applied", "1" },
                    { "active_id", "1234567812345678" },
                    { "effective_id", enabled ? "1234567812345678" : "0000000000000000" },
                    { "effective_controller", enabled ? "po4ao" : "integrator" },
                    { "reload", "idle" },
                    { "reload_error", "" } } ) );
    define( text( "po4ao", "fsm", { { "state", "READY" } } ) );
    define( runtimeSwitch( "control_enable", enabled ) );
    define( runtimeSwitch( "reset_model", false ) );
    define( runtimeSwitch( "recollect_warmup", false ) );
    define(
        text( "po4ao", "collection_status", { { "state", "ready" }, { "frames", "96" }, { "target_frames", "96" } } ) );
    m_publisher.handleSetProperty( text( "po4ao", "model_status", { { "sequence", "1" } } ) );
    drain();
}

Fixture::Fixture( bool companion )
{
    m_publisher.addSubscriber( &m_widget );
    m_publisher.onConnect();
    drain();
    if( companion )
        ready();
}

void Fixture::ready()
{
    define( text( "trainer", "fsm", { { "state", "READY" } } ) );
    define( text( "trainer", "devices", { { "nn", "nn_test" }, { "po4ao", "po4ao" } } ) );
    for( const auto *name : { "po4ao_enable", "reset", "recollect", "learning" } )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Switch, "trainer", name );
        property.setPerm( pcf::IndiProperty::ReadWrite );
        property.setState( pcf::IndiProperty::Ok );
        if( std::string( name ) == "learning" )
        {
            property.add( pcf::IndiElement( "run", pcf::IndiElement::Off ) );
            property.add( pcf::IndiElement( "pause", pcf::IndiElement::On ) );
        }
        else
            property.add( pcf::IndiElement(
                std::string( name ) == "reset" || std::string( name ) == "recollect" ? "request" : "toggle",
                pcf::IndiElement::Off ) );
        define( property );
    }
    nn();
    define( text( "trainer",
                  "status",
                  { { "heartbeat", "0" },
                    { "reconstructor", "cacao" },
                    { "controller", "integrator" },
                    { "reconstruction_phase", "idle" },
                    { "training_phase", "idle" },
                    { "learning", "idle" },
                    { "collection", "ready" },
                    { "modulation", "Off" },
                    { "active_policy", "0000000000000000" },
                    { "retained_policy", "1234" },
                    { "message", "Ready" },
                    { "error", "" } } ) );
    define( text( "trainer", "progress", { { "iteration", "0" } } ) );
    status( { { "heartbeat", "1" } } );
    trainingControls();
}

void Fixture::trainingControls()
{
    define( text( "trainer",
                  "policy_catalog",
                  { { "internal_a", "internal-bright | 2026-11-03 15:22 UTC | iteration 300" },
                    { "sky_b", "onsky | 2026-11-03 19:14 UTC | iteration 500" } } ) );
    define( text( "trainer", "set_policy", { { "id", "" } } ) );
    define( text( "trainer", "save_policy", { { "label", "" } } ) );
    for( const auto &entry : std::map<std::string, double>{ { "command_regularization", 0.01 },
                                                            { "delta_policy_regularization", 0.5 },
                                                            { "policy_learning_rate", 0.001 } } )
    {
        pcf::IndiProperty property( pcf::IndiProperty::Number, "trainer", entry.first );
        property.setPerm( pcf::IndiProperty::ReadWrite );
        property.setState( pcf::IndiProperty::Ok );
        property.add( pcf::IndiElement( "current", entry.second ) );
        property.add( pcf::IndiElement( "target", entry.second ) );
        define( property );
    }
    status( { { "save_available", "yes" }, { "warmup_required", "no" }, { "delta_policy_mode", "scalar" } } );
}

void Fixture::nn()
{
    define( text( "nn_test", "fsm", { { "state", "OPERATING" } } ) );
    define( text( "nn_test", "nn_activate", { { "model", "" } } ) );
    define( text( "nn_test", "reconstructor", { { "target", "" }, { "expected", "" } } ) );
    auto link = runtimeSwitch( "link_modulation", false );
    link.setDevice( "nn_test" );
    define( link );
    define( text( "nn_test",
                  "reconstruction_status",
                  { { "instance", "nn-1" },
                    { "heartbeat", "0" },
                    { "reconstructor", "cacao" },
                    { "phase", "idle" },
                    { "modulation", "Off" },
                    { "error", "" } } ) );
    pcf::IndiProperty catalog( pcf::IndiProperty::Switch, "nn_test", "model" );
    for( const auto &entry :
         { std::make_pair( "engine0", "model-a.trt" ), std::make_pair( "engine1", "model-b.trt" ) } )
    {
        pcf::IndiElement element( entry.first, pcf::IndiElement::Off );
        element.setLabel( entry.second );
        catalog.add( element );
    }
    define( catalog );
    define( text( "nn_test",
                  "model_status",
                  { { "requested", "model-a.trt" }, { "loaded", "model-a.trt" }, { "active", "" } } ) );
    m_publisher.handleSetProperty( text( "nn_test", "reconstruction_status", { { "heartbeat", "1" } } ) );
    drain();
}

void Fixture::define( const pcf::IndiProperty &property )
{
    m_publisher.handleDefProperty( property );
    drain();
}

void Fixture::status( std::initializer_list<std::pair<std::string, std::string>> elements )
{
    m_publisher.handleSetProperty( text( "trainer", "status", elements ) );
    for( const auto &entry : elements )
        if( entry.first == "reconstructor" || entry.first == "reconstruction_phase" )
            m_publisher.handleSetProperty(
                text( "nn_test",
                      "reconstruction_status",
                      { { entry.first == "reconstruction_phase" ? "phase" : entry.first, entry.second } } ) );
    drain();
}

void Fixture::runtimeStatus( std::initializer_list<std::pair<std::string, std::string>> elements )
{
    m_publisher.handleSetProperty( text( "po4ao", "model_status", elements ) );
    drain();
}

void Fixture::acknowledge( const std::string &name, pcf::IndiProperty::PropertyStateType state )
{
    const bool        nn = name == "nn_activate" || name == "reconstructor" || name == "link_modulation";
    pcf::IndiProperty property( pcf::IndiProperty::Text, nn ? "nn_test" : "trainer", name );
    property.setState( state );
    m_publisher.handleSetProperty( property );
    drain();
}

void Fixture::erase( const std::string &device, const std::string &name )
{
    m_publisher.handleDelProperty( text( device, name, {} ) );
    drain();
}

template <typename T>
T *Fixture::control( const char *name )
{
    auto *result = m_widget.findChild<T *>( name );
    REQUIRE( result != nullptr );
    return result;
}

void Fixture::complete( const std::string &name )
{
    acknowledge( name, pcf::IndiProperty::Busy );
    acknowledge( name, pcf::IndiProperty::Ok );
}
/// \endcond

/** \brief Catalog selection stays local and Enable sends the catalog identifier exactly once. */
TEST_CASE( "NN selection is staged until Enable", "[learningCtrl]" )
{
    Fixture f;
    auto   *combo = f.control<QComboBox>( "comboEngine" );
    REQUIRE( combo->count() == 2 );
    REQUIRE( combo->currentIndex() == -1 );
    combo->setCurrentIndex( 1 );
    REQUIRE( f.m_publisher.m_commands.empty() );
    f.status( { { "controller", "po4ao" },
                { "learning", "running" },
                { "collection", "collecting" },
                { "training_phase", "recollecting" } } );
    REQUIRE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
    f.m_widget.on_buttonEnableNN_clicked();
    f.m_widget.on_buttonEnableNN_clicked();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    const auto &command = f.m_publisher.m_commands.back();
    REQUIRE( command.getDevice() == "nn_test" );
    REQUIRE( command.getName() == "nn_activate" );
    REQUIRE( command["model"].get<std::string>() == "engine1" );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    f.acknowledge( "nn_activate", pcf::IndiProperty::Ok ); // Old Ok is not an acknowledgement.
    REQUIRE_FALSE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
    f.complete( "nn_activate" );
    REQUIRE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
    REQUIRE( combo->currentData().toString() == "engine1" );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::on_buttonEnableNN_clicked;
#endif
    // clang-format on
}

/** \brief Reconstructor requests specify the confirmed source and target, never the staged engine. */
TEST_CASE( "Reconstructor handover uses an explicit target", "[learningCtrl]" )
{
    Fixture f;
    f.control<QComboBox>( "comboEngine" )->setCurrentIndex( 1 );
    f.status( { { "learning", "resetting" }, { "training_phase", "resetting" } } );
    f.m_widget.on_buttonReconstructor_clicked();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    const auto command = f.m_publisher.m_commands.back();
    REQUIRE( command.getName() == "reconstructor" );
    REQUIRE( command["expected"].get<std::string>() == "cacao" );
    REQUIRE( command["target"].get<std::string>() == "nn" );
    REQUIRE_FALSE( command.find( "model" ) );
    f.status( { { "reconstructor", "nn" } } );
    f.complete( "reconstructor" );
    f.m_widget.on_buttonReconstructor_clicked();
    REQUIRE( f.m_publisher.m_commands.back()["target"].get<std::string>() == "cacao" );
}

/** \brief PO4AO and learning action labels follow readback rather than the request or retained ID. */
TEST_CASE( "Learning and PO4AO labels follow acknowledgements", "[learningCtrl]" )
{
    Fixture f;
    auto   *po = f.control<QPushButton>( "buttonPO4AO" );
    REQUIRE( po->text() == "Enable PO4AO" );
    f.m_widget.on_buttonPO4AO_clicked();
    REQUIRE( po->text() == "Enable PO4AO" );
    REQUIRE( f.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::On );
    f.status( { { "controller", "fallback" } } );
    f.complete( "po4ao_enable" );
    REQUIRE( po->text() == "Disable PO4AO" );
    f.status( { { "learning", "waiting" } } );
    auto *learning = f.control<QPushButton>( "buttonLearning" );
    REQUIRE( learning->text() == "Pause learning" );
    f.m_widget.on_buttonLearning_clicked();
    REQUIRE( f.m_publisher.m_commands.back()["pause"].getSwitchState() == pcf::IndiElement::On );
    REQUIRE( learning->text() == "Pause learning" );
    REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.status( { { "learning", "paused" } } );
    f.complete( "learning" );
    REQUIRE( learning->text() == "Resume learning" );
    f.m_widget.on_buttonLearning_clicked();
    REQUIRE( f.m_publisher.m_commands.back()["run"].getSwitchState() == pcf::IndiElement::On );
}

/** \brief Each domain sends requests only to its own backend. */
TEST_CASE( "Modulation and PO4AO requests use independent backends", "[learningCtrl]" )
{
    Fixture f;
    f.m_widget.on_checkModulation_clicked( true );
    REQUIRE( f.m_publisher.m_commands.back().getName() == "link_modulation" );
    REQUIRE_FALSE( f.control<QCheckBox>( "checkModulation" )->isChecked() );
    f.complete( "link_modulation" );
    f.m_widget.on_buttonWarmup_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getName() == "recollect" );
    f.m_widget.on_buttonReset_clicked(); // Reset remains available to cancel a running operation.
    REQUIRE( f.m_publisher.m_commands.back().getName() == "reset" );
    for( const auto &command : f.m_publisher.m_commands )
        REQUIRE( command.getDevice() == ( command.getName() == "link_modulation" ? "nn_test" : "trainer" ) );
}

/** \brief Unknown/conflicting reconstructor evidence is not interpreted as CACAO active. */
TEST_CASE( "Unknown controller state does not guess actions", "[learningCtrl]" )
{
    Fixture f;
    for( const auto *source : { "unknown", "conflict", "transition" } )
    {
        f.status( { { "reconstructor", source } } );
        REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
        f.m_widget.on_buttonReconstructor_clicked();
    }
    f.status( { { "learning", "error" }, { "controller", "unknown" } } );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    REQUIRE( f.m_publisher.m_commands.empty() );
}

/** \brief Sparse updates preserve selection; deletion removes capabilities until valid redefinition. */
TEST_CASE( "Catalog and device lifecycle preserve staged selection safely", "[learningCtrl]" )
{
    Fixture f;
    auto   *combo = f.control<QComboBox>( "comboEngine" );
    combo->setCurrentIndex( 1 );
    pcf::IndiProperty update( pcf::IndiProperty::Switch, "nn_test", "model" );
    update.add( pcf::IndiElement( "engine0", pcf::IndiElement::On ) );
    f.m_publisher.handleSetProperty( update );
    drain();
    REQUIRE( combo->count() == 2 );
    REQUIRE( combo->currentData().toString() == "engine1" );
    f.erase( "nn_test", "model" );
    REQUIRE( combo->count() == 0 );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
    f.define( update );
    REQUIRE( combo->count() == 1 );
    REQUIRE( combo->currentIndex() == -1 );
    f.erase( "trainer", "fsm" );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.define( text( "trainer", "fsm", { { "state", "READY" } } ) );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
}

/** \brief Optional trainer mapping changes and deletion cannot change NN subscriptions. */
TEST_CASE( "NN controls are independent of trainer discovery", "[learningCtrl][independentNN]" )
{
    Fixture f( false );
    f.nn();
    auto *combo = f.control<QComboBox>( "comboEngine" );
    combo->setCurrentIndex( 1 );
    REQUIRE( f.control<QLabel>( "labelNNTitle" )->text() == "nnReconstructor" );
    REQUIRE( f.control<QLabel>( "labelPO4AOTitle" )->text() == "PO4AO" );
    REQUIRE( f.control<QLabel>( "labelEngine" )->text() == "CNN Reconstructor:" );
    REQUIRE( f.control<QPushButton>( "buttonEnableNN" )->text() == "Set reconstructor" );
    REQUIRE( f.control<QPushButton>( "buttonReconstructor" )->text() == "Switch to CNN" );
    REQUIRE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    f.define( text( "trainer", "devices", { { "nn", "new_nn" } } ) );
    f.erase( "trainer", "" );
    REQUIRE( combo->count() == 2 );
    REQUIRE( combo->currentIndex() == 1 );
    f.m_widget.on_buttonEnableNN_clicked();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    REQUIRE( f.m_publisher.m_commands.back().getDevice() == "nn_test" );
    REQUIRE( f.m_publisher.m_commands.back().getName() == "nn_activate" );
    f.complete( "nn_activate" );
    f.status( { { "reconstructor", "nn" } } );
    REQUIRE( f.control<QPushButton>( "buttonReconstructor" )->text() == "Switch to CACAO" );
    f.m_widget.on_buttonReconstructor_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getDevice() == "nn_test" );
    REQUIRE( f.m_publisher.m_commands.back()["target"].get() == "cacao" );
}

/** \brief An NN restart invalidates NN controls without invalidating an available trainer. */
TEST_CASE( "Independent NN restart requires fresh capabilities", "[learningCtrl][independentNN]" )
{
    Fixture f;
    f.m_publisher.handleSetProperty(
        text( "nn_test", "reconstruction_status", { { "instance", "nn-2" }, { "heartbeat", "0" } } ) );
    drain();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    REQUIRE( f.control<QComboBox>( "comboEngine" )->count() == 0 );
    REQUIRE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.m_publisher.m_commands.empty() );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::observeHeartbeat;
    xqt::learningCtrl::handleDelProperty;
#endif
    // clang-format on
}

/** \brief Reconnection discards request state and never replays commands; read-only properties stay disabled. */
TEST_CASE( "Disconnect and permissions prevent stale commands", "[learningCtrl]" )
{
    Fixture f;
    f.m_widget.on_buttonLearning_clicked();
    f.m_widget.onDisconnect();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.m_widget.onConnect();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    auto command = text( "nn_test", "nn_activate", { { "model", "" } } );
    command.setPerm( pcf::IndiProperty::ReadOnly );
    f.define( command );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
}

/** \brief Missing acknowledgement remains visibly unknown and never resends the request. */
TEST_CASE( "Acknowledgement timeout does not imply remote completion", "[learningCtrl]" )
{
    Fixture f;
    f.m_widget.on_buttonReconstructor_clicked();
    f.m_widget.acknowledgementTimeout();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    REQUIRE( f.control<QLabel>( "labelStatus" )->text().contains( "unknown" ) );
    REQUIRE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    f.acknowledge( "reconstructor", pcf::IndiProperty::Alert );
    REQUIRE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
}

/** \brief Two clients send the same explicit destination and do not change each other's staged choices. */
TEST_CASE( "Two GUI clients share readbacks without toggling optimistically", "[learningCtrl]" )
{
    Fixture           first;
    xqt::learningCtrl second( "trainer", nullptr, "po4ao", "nn_test" );
    first.m_publisher.addSubscriber( &second );
    first.m_publisher.onConnect();
    drain();
    first.ready();
    first.control<QComboBox>( "comboEngine" )->setCurrentIndex( 1 );
    first.m_widget.on_buttonReconstructor_clicked();
    second.on_buttonReconstructor_clicked();
    REQUIRE( first.m_publisher.m_commands.size() == 2 );
    REQUIRE( first.m_publisher.m_commands[0]["target"].get() == first.m_publisher.m_commands[1]["target"].get() );
    REQUIRE( second.findChild<QComboBox *>( "comboEngine" )->currentIndex() == -1 );
    REQUIRE( first.control<QLabel>( "labelReconstructor" )->text() == "Active: CACAO" );
}

/** \brief A confirmed stopped/faulted NN permits recovery through either reconstruction control. */
TEST_CASE( "Stopped writers expose explicit recovery", "[learningCtrl]" )
{
    Fixture f;
    f.control<QComboBox>( "comboEngine" )->setCurrentIndex( 1 );
    f.status( { { "reconstructor", "stopped" }, { "reconstruction_phase", "error" } } );
    REQUIRE( f.control<QPushButton>( "buttonEnableNN" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonReconstructor" )->text() == "Switch to CACAO" );
    f.m_widget.on_buttonReconstructor_clicked();
    REQUIRE( f.m_publisher.m_commands.back()["expected"].get() == "stopped" );
    REQUIRE( f.m_publisher.m_commands.back()["target"].get() == "cacao" );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::updateGUI;
    xqt::learningCtrl::on_buttonReconstructor_clicked;
#endif
    // clang-format on
}

/** \brief Disable can preserve a session during publication/recollection; Reset remains a separate action. */
TEST_CASE( "Disable supersedes busy training operations", "[learningCtrl]" )
{
    for( const auto *phase : { "publishing", "recollecting" } )
    {
        Fixture f;
        f.status( { { "controller", "po4ao" },
                    { "training_phase", phase },
                    { "collection", "collecting" },
                    { "collection_frames", "21" },
                    { "collection_target", "100" } } );
        f.acknowledge( "recollect", pcf::IndiProperty::Busy );
        REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        REQUIRE( f.control<QLabel>( "labelCollection" )->text().contains( "21/100" ) );
        f.m_widget.on_buttonPO4AO_clicked();
        REQUIRE( f.m_publisher.m_commands.back().getName() == "po4ao_enable" );
        REQUIRE( f.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    }
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::updateGUI;
    xqt::learningCtrl::on_buttonPO4AO_clicked;
#endif
    // clang-format on
}

/** \brief Cached/unchanged heartbeat values cannot keep a silent companion actionable. */
TEST_CASE( "Companion status expires until heartbeat advances", "[learningCtrl]" )
{
    Fixture f;
    QThread::msleep( 5100 );
    f.status( { { "heartbeat", "1" } } );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    REQUIRE( f.control<QLabel>( "labelConnection" )->text().contains( "Stale" ) );
    REQUIRE( f.control<QLabel>( "labelReconstructor" )->text().contains( "Unknown" ) );
    f.status( { { "heartbeat", "2" } } );
    REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    f.define(
        text( "nn_test",
              "reconstruction_status",
              { { "instance", "nn-1" }, { "heartbeat", "1" }, { "reconstructor", "cacao" }, { "phase", "idle" } } ) );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    f.m_publisher.handleSetProperty( text( "nn_test", "reconstruction_status", { { "heartbeat", "2" } } ) );
    drain();
    REQUIRE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    f.erase( "trainer", "status" );
    f.define( text( "trainer", "status", { { "heartbeat", "2" } } ) );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.status( { { "heartbeat", "3" } } );
    REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    REQUIRE( f.m_publisher.m_commands.empty() );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::observeHeartbeat;
    xqt::learningCtrl::updateGUI;
#endif
    // clang-format on
}

/** \brief Config errors remain visible and long engine names stay accessible in a compact layout. */
TEST_CASE( "Compact layout retains long labels and configuration errors", "[learningCtrl]" )
{
    Fixture           f;
    pcf::IndiProperty catalog( pcf::IndiProperty::Switch, "nn_test", "model" );
    pcf::IndiElement  engine( "engine_long", pcf::IndiElement::Off );
    const std::string label( 180, 'x' );
    engine.setLabel( label + ".trt" );
    catalog.add( engine );
    f.define( catalog );
    auto *combo = f.control<QComboBox>( "comboEngine" );
    combo->setCurrentIndex( 0 );
    REQUIRE( combo->toolTip() == QString::fromStdString( label + ".trt" ) );
    f.define( text( "trainer",
                    "training_config",
                    { { "error", "Invalid learning rate; previous configuration remains active" } } ) );
    f.m_widget.resize( 500, 330 );
    f.m_widget.show();
    drain();
    REQUIRE( f.m_widget.width() == 500 );
    INFO( f.control<QLabel>( "labelStatus" )->text().toStdString() );
    REQUIRE( f.control<QLabel>( "labelStatus" )->text().contains( "Invalid learning rate" ) );
    for( auto *button : f.m_widget.findChildren<QPushButton *>() )
        REQUIRE( f.m_widget.rect().contains( QRect( button->mapTo( &f.m_widget, QPoint() ), button->size() ) ) );
    f.m_widget.resize( 420, 330 );
    drain();
    REQUIRE( f.m_widget.width() == 420 );
    for( auto *button : f.m_widget.findChildren<QPushButton *>() )
        REQUIRE( f.m_widget.rect().contains( QRect( button->mapTo( &f.m_widget, QPoint() ), button->size() ) ) );
}

/** \brief Native inference controls work with no trainer definitions or process. */
TEST_CASE( "PO4AO inference is independent of the optional trainer", "[learningCtrl]" )
{
    Fixture f( false );
    f.runtime();
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonWarmup" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
    f.m_widget.on_buttonLearning_clicked();
    REQUIRE( f.m_publisher.m_commands.empty() );
    f.m_widget.on_buttonPO4AO_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getDevice() == "po4ao" );
    REQUIRE( f.m_publisher.m_commands.back().getName() == "control_enable" );
    REQUIRE( f.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::On );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->text() == "Enable PO4AO" );
    f.define( runtimeSwitch( "control_enable", false, pcf::IndiProperty::Busy ) );
    f.define( runtimeSwitch( "control_enable", true ) );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->text() == "Disable PO4AO" );
    f.m_widget.on_buttonWarmup_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getName() == "recollect_warmup" );
    f.define( runtimeSwitch( "recollect_warmup", false, pcf::IndiProperty::Busy ) );
    f.m_publisher.handleSetProperty(
        text( "po4ao", "collection_status", { { "state", "collecting" }, { "frames", "12" } } ) );
    drain();
    REQUIRE( f.control<QLabel>( "labelCollection" )->text().contains( "12/96" ) );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    f.m_widget.on_buttonPO4AO_clicked();
    REQUIRE( f.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::Off );
    for( const auto &command : f.m_publisher.m_commands )
        REQUIRE( command.getDevice() == "po4ao" );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::updateRuntimeGUI;
    xqt::learningCtrl::on_buttonPO4AO_clicked;
    xqt::learningCtrl::on_buttonWarmup_clicked;
    xqt::learningCtrl::on_buttonLearning_clicked;
#endif
    // clang-format on
}

/** \brief Direct Reset selects integrator when needed and waits for native acknowledgements. */
TEST_CASE( "Native Reset leaves notebook ownership with the operator", "[learningCtrl]" )
{
    for( const bool enabled : { false, true } )
    {
        Fixture f( false );
        f.runtime( enabled );
        f.m_widget.on_buttonReset_clicked();
        REQUIRE( f.m_publisher.m_commands.size() == 2 );
        REQUIRE( f.m_publisher.m_commands.back().getName() == "reset_model" );
        REQUIRE( f.m_publisher.m_commands.back()["toggle"].getSwitchState() == pcf::IndiElement::On );
        if( enabled )
        {
            REQUIRE( f.m_publisher.m_commands.front().getName() == "control_enable" );
            REQUIRE( f.m_publisher.m_commands.front()["toggle"].getSwitchState() == pcf::IndiElement::Off );
            f.define( runtimeSwitch( "control_enable", true, pcf::IndiProperty::Busy ) );
            f.define( runtimeSwitch( "control_enable", false ) );
        }
        REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        f.define( runtimeSwitch( "reset_model", false, pcf::IndiProperty::Busy ) );
        f.define( runtimeSwitch( "reset_model", false ) );
        REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        f.runtimeStatus( { { "reset_request", "2" },
                           { "reset_applied", "2" },
                           { "control_target", "off" },
                           { "control_request", enabled ? "2" : "1" },
                           { "control_applied", enabled ? "2" : "1" } } );
        REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        for( const auto &command : f.m_publisher.m_commands )
            REQUIRE( command.getDevice() == "po4ao" );
    }
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::on_buttonReset_clicked;
    xqt::learningCtrl::acknowledge;
#endif
    // clang-format on
}

/** \brief The same window uses managed coordination after discovery without replaying native requests. */
TEST_CASE( "Trainer arrival and departure do not require another GUI", "[learningCtrl]" )
{
    Fixture f( false );
    f.runtime();
    f.m_widget.on_buttonWarmup_clicked();
    f.ready();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    f.define( runtimeSwitch( "recollect_warmup", false, pcf::IndiProperty::Busy ) );
    f.define( runtimeSwitch( "recollect_warmup", false ) );
    REQUIRE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    f.m_widget.on_buttonLearning_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getDevice() == "trainer" );
    REQUIRE( f.m_publisher.m_commands.back().getName() == "learning" );
    f.complete( "learning" );
    f.m_widget.on_buttonPO4AO_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getName() == "po4ao_enable" );
    f.complete( "po4ao_enable" );
    f.erase( "trainer", "" );
    REQUIRE( f.m_publisher.m_commands.size() == 3 );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    f.m_widget.on_buttonPO4AO_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getDevice() == "po4ao" );
    REQUIRE( f.m_publisher.m_commands.back().getName() == "control_enable" );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::trainerReady;
    xqt::learningCtrl::handleDelProperty;
    xqt::learningCtrl::updateGUI;
#endif
    // clang-format on
}

/** \brief Runtime permissions, deletion and restart never inherit previous-process capabilities. */
TEST_CASE( "Native PO4AO discovery and restart invalidate stale state", "[learningCtrl]" )
{
    Fixture f( false );
    f.runtime();
    auto control = runtimeSwitch( "control_enable", false );
    control.setPerm( pcf::IndiProperty::ReadOnly );
    f.define( control );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    f.erase( "po4ao", "recollect_warmup" );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonWarmup" )->isEnabled() );
    f.m_publisher.handleSetProperty(
        text( "po4ao", "model_status", { { "instance", "runtime-2" }, { "sequence", "0" } } ) );
    drain();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.m_publisher.handleSetProperty( text( "po4ao", "model_status", { { "sequence", "1" } } ) );
    drain();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.runtime( false, "runtime-2" );
    REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.m_widget.onDisconnect();
    f.m_widget.onConnect();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    REQUIRE( f.m_publisher.m_commands.empty() );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::observeHeartbeat;
    xqt::learningCtrl::commandAvailable;
    xqt::learningCtrl::onDisconnect;
#endif
    // clang-format on
}

/** \brief Independent heartbeats permit native control after trainer loss, without rerouting an unresolved action. */
TEST_CASE( "Trainer expiry preserves independent runtime control", "[learningCtrl]" )
{
    Fixture f;
    f.runtime();
    QThread::msleep( 5100 );
    f.m_publisher.handleSetProperty( text( "po4ao", "model_status", { { "sequence", "1" } } ) );
    drain();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    f.m_publisher.handleSetProperty( text( "po4ao", "model_status", { { "sequence", "2" } } ) );
    drain();
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.control<QLabel>( "labelConnection" )->text().contains( "trainer unavailable" ) );
    f.status( { { "heartbeat", "2" } } );
    REQUIRE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    f.m_widget.on_buttonReset_clicked();
    f.erase( "trainer", "status" );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    f.m_widget.on_buttonPO4AO_clicked();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    f.complete( "reset" );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::observeHeartbeat;
    xqt::learningCtrl::updateRuntimeGUI;
    xqt::learningCtrl::acknowledge;
#endif
    // clang-format on
}

/** \brief Another client's managed operation remains unresolved after trainer heartbeat expiry. */
TEST_CASE( "Trainer expiry cannot bypass an observed Busy operation", "[learningCtrl][review]" )
{
    Fixture f;
    f.runtime();
    SECTION( "A command from another client" )
    {
        f.acknowledge( "recollect", pcf::IndiProperty::Busy );
    }
    SECTION( "Automatic publication has no command vector" )
    {
        f.status( { { "training_phase", "publishing" } } );
    }
    QThread::msleep( 5100 );
    f.runtimeStatus( { { "sequence", "2" } } );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonWarmup" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.m_widget.on_buttonWarmup_clicked();
    REQUIRE( f.m_publisher.m_commands.empty() );
    f.acknowledge( "recollect", pcf::IndiProperty::Ok );
    f.status( { { "training_phase", "idle" } } );
    REQUIRE( f.control<QPushButton>( "buttonWarmup" )->isEnabled() );
    f.m_widget.on_buttonWarmup_clicked();
    REQUIRE( f.m_publisher.m_commands.back().getDevice() == "po4ao" );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::updateGUI;
    xqt::learningCtrl::updateRoute;
    xqt::learningCtrl::updateRuntimeGUI;
#endif
    // clang-format on
}

/** \brief Trainer arrival cannot bypass another native client; normal managed supersession remains available. */
TEST_CASE( "Trainer arrival preserves outstanding native operations", "[learningCtrl][review]" )
{
    Fixture f( false );
    f.runtime();
    f.define( runtimeSwitch( "recollect_warmup", false, pcf::IndiProperty::Busy ) );
    f.ready();
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    f.m_widget.on_buttonReset_clicked();
    REQUIRE( f.m_publisher.m_commands.empty() );
    SECTION( "Completion releases the route" )
    {
        f.define( runtimeSwitch( "recollect_warmup", false ) );
    }
    SECTION( "Device invalidation releases the route" )
    {
        f.erase( "po4ao", "" );
    }
    REQUIRE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    // Native work issued during the current managed route belongs to its existing coordination.
    f.define( runtimeSwitch( "control_enable", true, pcf::IndiProperty::Busy ) );
    f.status( { { "controller", "po4ao" }, { "training_phase", "publishing" } } );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::updateGUI;
    xqt::learningCtrl::updateRoute;
    xqt::learningCtrl::handleDelProperty;
#endif
    // clang-format on
}

/** \brief Reset cancels pending Enable without frames; generation readback settles superseded/no-op selections. */
TEST_CASE( "Native Reset supersedes Enable awaiting input frames", "[learningCtrl][review]" )
{
    for( const bool observed : { false, true } )
    {
        Fixture f( false );
        f.runtime();
        f.m_widget.on_buttonPO4AO_clicked();
        if( observed )
        {
            f.runtimeStatus( { { "control_target", "on" }, { "control_request", "2" } } );
            f.define( runtimeSwitch( "control_enable", false, pcf::IndiProperty::Busy ) );
        }
        REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
        f.m_widget.on_buttonReset_clicked();
        REQUIRE( f.m_publisher.m_commands.size() == 3 );
        REQUIRE( f.m_publisher.m_commands[1].getName() == "control_enable" );
        REQUIRE( f.m_publisher.m_commands[1]["toggle"].getSwitchState() == pcf::IndiElement::Off );
        REQUIRE( f.m_publisher.m_commands[2].getName() == "reset_model" );
        f.runtimeStatus( { { "control_target", "off" }, { "control_request", "3" }, { "reset_request", "2" } } );
        // Cached switch Ok alone cannot finish Reset; a Busy transition may have been coalesced.
        f.define( runtimeSwitch( "reset_model", false ) );
        REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        f.runtimeStatus( { { "reset_applied", "2" } } );
        REQUIRE_FALSE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        f.define( runtimeSwitch( "control_enable", false ) );
        f.runtimeStatus( { { "control_applied", "3" } } );
        REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
        REQUIRE( f.control<QPushButton>( "buttonReset" )->isEnabled() );
        REQUIRE( f.m_publisher.m_commands.size() == 3 );
    }
    // clang-format off
#ifdef LEARNINGCTRL_TEST_DOXYGEN_REF
    xqt::learningCtrl::on_buttonReset_clicked;
    xqt::learningCtrl::acknowledge;
#endif
    // clang-format on
}

/** \brief Numeric Enter sends only the target and preserves current readback until acknowledgement. */
TEST_CASE( "Trainer numeric targets retain current readback", "[learningCtrl]" )
{
    Fixture f;
    auto   *entry = f.control<xqt::statusLineEdit>( "entryLearningRate" );
    REQUIRE( entry->isEnabled() );
    REQUIRE( entry->currText().toDouble() == Approx( 0.001 ) );
    entry->setEditText( "1e-7" );
    REQUIRE( QMetaObject::invokeMethod( entry, "returnPressed", Qt::DirectConnection ) );
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    const auto &sent = f.m_publisher.m_commands.back();
    REQUIRE( sent.getType() == pcf::IndiProperty::Number );
    REQUIRE( sent.getName() == "policy_learning_rate" );
    REQUIRE( sent.getElements().size() == 1 );
    REQUIRE( sent["target"].get<double>() == Approx( 1e-7 ) );
    pcf::IndiProperty update( pcf::IndiProperty::Number, "trainer", "policy_learning_rate" );
    update.setState( pcf::IndiProperty::Busy );
    update.add( pcf::IndiElement( "target", 1e-7 ) );
    f.m_publisher.handleSetProperty( update );
    drain();
    REQUIRE( entry->currText().toDouble() == Approx( 0.001 ) );
    REQUIRE( entry->toolTip().contains( "pending" ) );
    update.setState( pcf::IndiProperty::Ok );
    update.add( pcf::IndiElement( "current", 1e-7 ) );
    f.m_publisher.handleSetProperty( update );
    drain();
    REQUIRE( entry->currText().toDouble() == Approx( 1e-7 ) );
    REQUIRE_FALSE( entry->toolTip().contains( "pending" ) );
    entry->setEditText( "0" );
    QMetaObject::invokeMethod( entry, "returnPressed", Qt::DirectConnection );
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    f.status( { { "delta_policy_mode", "per_mode" } } );
    REQUIRE_FALSE( f.control<xqt::statusLineEdit>( "entryDeltaRegularization" )->isEnabled() );
    REQUIRE( f.control<xqt::statusLineEdit>( "entryDeltaRegularization" )->text().contains( "Per-mode" ) );
    REQUIRE( entry->isEnabled() );
    f.erase( "trainer", "" );
    f.runtime( true );
    REQUIRE_FALSE( entry->isEnabled() );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonSetPolicy" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonPO4AO" )->isEnabled() );
    REQUIRE( f.control<QPushButton>( "buttonReconstructor" )->isEnabled() );
}

/** \brief Stable saved IDs survive redefinition; selection alone cannot load or resume training. */
TEST_CASE( "Saved pairs use staged stable catalog selection", "[learningCtrl]" )
{
    Fixture f;
    auto   *combo = f.control<QComboBox>( "comboPolicy" );
    REQUIRE( combo->currentIndex() == -1 );
    combo->setCurrentIndex( combo->findData( "internal_a" ) );
    REQUIRE( f.m_publisher.m_commands.empty() );
    f.erase( "trainer", "policy_catalog" );
    f.define( text( "trainer", "policy_catalog", { { "aaa", "Other" }, { "internal_a", "New display label" } } ) );
    REQUIRE( combo->currentData().toString() == "internal_a" );
    REQUIRE( f.m_publisher.m_commands.empty() );
    f.m_widget.on_buttonSetPolicy_clicked();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    REQUIRE( f.m_publisher.m_commands.back().getName() == "set_policy" );
    REQUIRE( f.m_publisher.m_commands.back()["id"].get<std::string>() == "internal_a" );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    f.complete( "set_policy" );
    f.status( { { "learning", "paused" }, { "warmup_required", "yes" } } );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.control<QLabel>( "labelCollection" )->text() == "Fresh warmup required" );
    REQUIRE( f.control<QPushButton>( "buttonWarmup" )->isEnabled() );
    f.status( { { "warmup_required", "no" } } );
    REQUIRE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    f.status( { { "initialization_required", "yes" } } );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonLearning" )->isEnabled() );
    REQUIRE( f.control<QLabel>( "labelLearning" )->text() == "Set policy or Reset required" );
}

/** \brief Save remains available during publication and transmits only the operator's optional label. */
TEST_CASE( "Saving a pair does not pause a running publication", "[learningCtrl]" )
{
    Fixture f;
    f.status( { { "training_phase", "publishing" }, { "learning", "running" } } );
    REQUIRE( f.control<QPushButton>( "buttonSavePolicy" )->isEnabled() );
    QTimer::singleShot( 0,
                        []()
                        {
                            for( auto *widget : QApplication::topLevelWidgets() )
                                if( auto *dialog = qobject_cast<QInputDialog *>( widget ) )
                                {
                                    dialog->setTextValue( "internal-bright" );
                                    dialog->accept();
                                }
                        } );
    f.m_widget.on_buttonSavePolicy_clicked();
    REQUIRE( f.m_publisher.m_commands.size() == 1 );
    const auto &sent = f.m_publisher.m_commands.back();
    REQUIRE( sent.getName() == "save_policy" );
    REQUIRE( sent["label"].get<std::string>() == "internal-bright" );
    f.complete( "save_policy" );
    f.status( { { "save_available", "no" } } );
    REQUIRE_FALSE( f.control<QPushButton>( "buttonSavePolicy" )->isEnabled() );
}

/** \brief Save a representative real-widget preview without connecting an INDI client. */
TEST_CASE( "Save the compact panel preview", "[.preview]" )
{
    Fixture f;
    f.control<QComboBox>( "comboEngine" )->setCurrentIndex( 0 );
    f.status( { { "reconstructor", "nn" },
                { "controller", "po4ao" },
                { "learning", "running" },
                { "active_policy", "a31c480de728b941" },
                { "retained_policy", "a31c480de728b941" } } );
    f.define( text( "trainer", "progress", { { "iteration", "250" } } ) );
    if( std::getenv( "LEARNINGCTRL_NOTEBOOK_PREVIEW" ) )
    {
        f.erase( "trainer", "" );
        f.runtime( true );
    }
    f.control<QComboBox>( "comboPolicy" )->setCurrentIndex( 0 );
    f.m_widget.resize( 540, 410 );
    f.m_widget.show();
    drain();
    REQUIRE( f.m_widget.width() == 540 );
    const char *path = std::getenv( "LEARNINGCTRL_SCREENSHOT" );
    REQUIRE( path != nullptr );
    REQUIRE( f.m_widget.grab().save( path ) );
}

} // namespace learningCtrlGUITest
} // namespace libXWCTest

/// Run widget tests with the normal stylesheet and no network connection.
int main( int   argc, /**< [in] Test argument count. */
          char *argv[] /**< [in] Catch2 arguments. */ )
{
    QApplication app( argc, argv );
    QFile        style( ":/magaox.qss" );
    if( style.open( QFile::ReadOnly | QFile::Text ) )
    {
        QTextStream stream( &style );
        app.setStyleSheet( stream.readAll() );
    }
    return Catch::Session().run( argc, argv );
}
