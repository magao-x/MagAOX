/** \file stagePairStatus.hpp
 * \brief Shared preset selector for a pair of standard motion stages.
 */

#ifndef stagePairStatus_hpp
#define stagePairStatus_hpp

#include <array>
#include <map>

#include "statusCombo.hpp"

namespace xqt
{

/// Select a named preset shared by two stages using the standard status-combo controls.
/** Definitions replace each stage's choices; partial updates merge switch states.
 * A displayed preset means both stages report that name, which can acknowledge a
 * target during motion. Commands require both stages READY and are sent separately
 * to each controller; the protocol does not provide an atomic paired move.
 */
class stagePairStatus : public statusCombo
{
    Q_OBJECT

  protected:
    /// Observations belonging to one independently subscribed stage.
    struct stageState
    {
        std::string device; ///< Stage device providing the observations.

        std::string fsm; ///< Last observed FSM state; empty while unknown.

        std::map<std::string, pcf::IndiElement::SwitchStateType> presets; ///< Observed preset names and switch states.
    };

    std::array<stageState, 2> m_stages; ///< Stage observations retained independently across property loss.

    bool m_connected{ false }; ///< Whether observations belong to an active shared connection.

  public:
    /// Construct an unconfigured paired-stage selector.
    stagePairStatus( QWidget        *Parent = nullptr, /**< [in] Owning Qt parent. */
                     Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */ );

    /// Configure the two stage devices and the selector label.
    void setup( const std::string &first,  /**< [in] First stage device. */
                const std::string &second, /**< [in] Second stage device. */
                const std::string &label /**< [in] Label beside the preset dropdown. */ );

    /// Subscribe independently to each stage's FSM and named presets.
    void subscribe() override;

    /// Accept observations on the active shared connection.
    void onConnect() override;

    /// Clear both stage observations and any staged or pending selection.
    void onDisconnect() override;

    /// Replace the corresponding stage observation before refreshing common presets.
    void handleDefProperty( const pcf::IndiProperty &property /**< [in] Received definition. */ ) override;

    /// Merge partial stage observations before refreshing common presets.
    void handleSetProperty( const pcf::IndiProperty &property /**< [in] Received update. */ ) override;

    /// Invalidate only the affected stage, FSM, or preset property.
    void
    handleDelProperty( const pcf::IndiProperty &property /**< [in] Deleted property or entire device. */ ) override;

    /// Display the common active preset or an unknown-position placeholder.
    QString formatValue() override;

    /// Send a validated common selection to each stage using its own device name.
    void sendNewProperty( const pcf::IndiProperty &property /**< [in] Common preset switch request. */ ) override;

  public slots:
    /// Refresh readiness and the standard status display without sending commands.
    void updateGUI() override;

    /// Submit the staged selection only while both stages are READY.
    void on_buttonGo_pressed() override;

  protected:
    /// Apply a definition or partial update to one stage's observations.
    void receiveProperty( const pcf::IndiProperty &property, /**< [in] Received stage property. */
                          bool definition /**< [in] Whether to replace the previous observation. */ );

    /// Rebuild the exact-name intersection and the common active-preset display.
    void refreshPresets();

    /// Cancel unsent edits and pending acknowledgment without scheduling commands.
    void clearSelection();
};

stagePairStatus::stagePairStatus( QWidget *Parent, Qt::WindowFlags f ) : statusCombo( Parent, f )
{
}

void stagePairStatus::setup( const std::string &first, const std::string &second, const std::string &label )
{
    m_stages[0].device = first;
    m_stages[1].device = second;
    statusCombo::setup( first, "presetName", "", label, "" );
    ctrlWidget( nullptr );
}

void stagePairStatus::subscribe()
{
    if( !m_parent )
        return;
    for( const auto &stage : m_stages )
    {
        m_parent->addSubscriberProperty( this, stage.device, "fsm" );
        m_parent->addSubscriberProperty( this, stage.device, "presetName" );
    }
}

void stagePairStatus::onConnect()
{
    m_connected = true;
    statusCombo::onConnect();
    updateGUI();
}

void stagePairStatus::onDisconnect()
{
    m_connected = false;
    for( auto &stage : m_stages )
    {
        stage.fsm.clear();
        stage.presets.clear();
    }
    statusCombo::onDisconnect();
    updateGUI();
}

void stagePairStatus::handleDefProperty( const pcf::IndiProperty &property )
{
    receiveProperty( property, true );
}

void stagePairStatus::handleSetProperty( const pcf::IndiProperty &property )
{
    receiveProperty( property, false );
}

void stagePairStatus::receiveProperty( const pcf::IndiProperty &property, bool definition )
{
    if( !m_connected )
        return;
    for( auto &stage : m_stages )
    {
        if( property.getDevice() != stage.device )
            continue;
        if( property.getName() == "fsm" )
        {
            if( definition || property.getType() != pcf::IndiProperty::Text )
                stage.fsm.clear();
            if( property.getType() == pcf::IndiProperty::Text && property.find( "state" ) )
                stage.fsm = property["state"].get();
        }
        else if( property.getName() == "presetName" )
        {
            if( definition || property.getType() != pcf::IndiProperty::Switch )
                stage.presets.clear();
            if( property.getType() == pcf::IndiProperty::Switch )
            {
                for( const auto &element : property.getElements() )
                {
                    auto state = element.second.getSwitchState();
                    if( state == pcf::IndiElement::On || state == pcf::IndiElement::Off )
                        stage.presets[element.first] = state;
                    else
                        stage.presets.erase( element.first );
                }
            }
        }
        else
            return;
        refreshPresets();
        return;
    }
}

void stagePairStatus::handleDelProperty( const pcf::IndiProperty &property )
{
    for( auto &stage : m_stages )
    {
        if( property.getDevice() != stage.device )
            continue;
        if( property.getName().empty() || property.getName() == "fsm" )
        {
            stage.fsm.clear();
            stage.presets.clear();
        }
        else if( property.getName() == "presetName" )
            stage.presets.clear();
        else
            return;
        clearSelection();
        refreshPresets();
        return;
    }
}

void stagePairStatus::clearSelection()
{
    m_statusEditing        = STOPPED;
    m_statusCommandPending = false;
    m_statusRequested.clear();
    m_statusEditTimer->stop();
    ui.status->setCurrentIndex( -1 );
    m_valChanged = true;
}

void stagePairStatus::refreshPresets()
{
    QStringList choices;
    for( const auto &preset : m_stages[0].presets )
    {
        if( !preset.first.empty() && preset.first != "none" && m_stages[1].presets.count( preset.first ) )
            choices.push_back( QString::fromStdString( preset.first ) );
    }
    QStringList previous;
    for( int index = 0; index < ui.status->count(); ++index )
        previous.push_back( ui.status->itemText( index ) );
    if( choices != previous )
    {
        const auto selection = ui.status->currentText();
        ui.status->clear();
        ui.status->addItems( choices );
        if( !selection.isEmpty() && choices.contains( selection ) )
            ui.status->setCurrentText( selection );
        else
            clearSelection();
    }

    const auto active = []( const stageState &stage )
    {
        std::string value;
        for( const auto &preset : stage.presets )
        {
            if( preset.second == pcf::IndiElement::On )
            {
                if( !value.empty() || preset.first.empty() )
                    return std::string();
                value = preset.first;
            }
        }
        return value;
    };
    m_value = active( m_stages[0] );
    if( m_value != active( m_stages[1] ) || !choices.contains( QString::fromStdString( m_value ) ) )
        m_value.clear();
    if( m_statusCommandPending && m_value == m_statusRequested )
        clearSelection();
    m_valChanged = true;
    updateGUI();
}

QString stagePairStatus::formatValue()
{
    return m_value.empty() ? QString( "---" ) : statusCombo::formatValue();
}

void stagePairStatus::updateGUI()
{
    const bool ready =
        m_connected && m_stages[0].fsm == "READY" && m_stages[1].fsm == "READY" && ui.status->count() > 0;
    setEnabled( ready );
    m_showVal = true;
    if( !ready && !m_statusCommandPending )
    {
        clearSelection();
        ui.status->setPlaceholderText( formatValue() );
    }
    statusCombo::updateGUI();
}

void stagePairStatus::on_buttonGo_pressed()
{
    updateGUI();
    if( !isEnabled() )
        return;
    statusCombo::on_buttonGo_pressed();
}

void stagePairStatus::sendNewProperty( const pcf::IndiProperty &property )
{
    if( !m_connected || m_stages[0].fsm != "READY" || m_stages[1].fsm != "READY" ||
        property.getName() != "presetName" || property.getType() != pcf::IndiProperty::Switch )
        return;
    std::string selection;
    for( const auto &element : property.getElements() )
    {
        if( element.second.getSwitchState() == pcf::IndiElement::On )
        {
            if( !selection.empty() )
                return;
            selection = element.first;
        }
    }
    if( selection.empty() || selection == "none" || !m_stages[0].presets.count( selection ) ||
        !m_stages[1].presets.count( selection ) )
        return;
    for( const auto &stage : m_stages )
    {
        pcf::IndiProperty command( pcf::IndiProperty::Switch, stage.device, "presetName" );
        command.setPerm( pcf::IndiProperty::ReadWrite );
        command.setState( pcf::IndiProperty::Idle );
        command.setRule( pcf::IndiProperty::OneOfMany );
        command.add( pcf::IndiElement( selection, pcf::IndiElement::On ) );
        xWidget::sendNewProperty( command );
    }
}

} // namespace xqt

#include "moc_stagePairStatus.cpp"

#endif
