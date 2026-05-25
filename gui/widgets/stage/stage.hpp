
/** \file stage.hpp
 * \brief Stage control widget used by stage and camera GUIs.
 */

#ifndef stage_hpp
#define stage_hpp

#include "ui_stage.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

/// Stage control widget with parked-poweroff display support.
class stage : public xWidget
{
    Q_OBJECT

    enum editchanges
    {
        NOTEDITING,
        STARTED,
        STOPPED
    };

  protected:
    /// Most recent FSM state reported by the stage controller.
    std::string m_appState;

    /// INDI device name of the stage controller.
    std::string m_stageName;
    /// Base window title shown while connected.
    std::string m_winTitle;

    /// Known preset names exposed by the controller.
    std::vector<std::string> m_presets;
    /// Placeholder for the current preset selection.
    std::string m_presetCurrent;
    /// Placeholder for the target preset selection.
    std::string m_presetTarget;

    /// Whether this widget is presenting a filter wheel style interface.
    bool m_filterWheel{ false };

    /// Current preset/filter label shown in the combo box.
    std::string m_setPoint;

    /// Whether the controller reports this stage as parked.
    bool m_parked{ false };

    /// Maximum allowed position in user units.
    double m_maxPos{ 100 };
    /// Current position in user units.
    double m_position{ -1e30 };
    /// Tracks when the displayed position should be highlighted as changed.
    bool m_position_changed{ false };

    /// Step size used by the incremental move buttons.
    double m_step{ 1 };

    /// Tracks edit state for the preset combo box.
    int m_setPointEditing{ STOPPED };
    /// Whether a requested preset/filter change is still awaiting live confirmation.
    bool m_setPointCommandPending{ false };
    /// Last preset/filter name sent from the combo box.
    std::string m_setPointRequested;
    /// Timer that clears staged preset edits after inactivity.
    QTimer *m_setPointEditTimer{ nullptr };

  public:
    /// Construct a stage widget for one INDI stage device.
    explicit stage( const std::string &stageName /**< [in] INDI device name for the controlled stage */,
                    QWidget           *Parent /**< [in] owning Qt parent widget */   = 0,
                    Qt::WindowFlags    f /**< [in] Qt window flags for the widget */ = Qt::WindowFlags() );

    /// Destroy the widget.
    ~stage();

    /// Subscribe to the stage properties needed by the widget.
    void subscribe();

    /// Reset the widget for an active INDI connection.
    virtual void onConnect();
    /// Reset the widget for a disconnected INDI connection.
    virtual void onDisconnect();

    /// Handle a newly defined property using the same path as updates.
    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Handle a deleted stage property.
    void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been deleted*/ );

    /// Handle a stage property update.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Clear focus from the widget.
    void clear_focus();

  public slots:
    /// Refresh the widget state from cached INDI values.
    void updateGUI();

    /// Mark the set-point combo box as actively edited.
    void on_setPoint_activated( int index /**< [in] combo-box index selected by the user */ );

    /// Send the currently selected preset/filter target.
    void on_setPointGo_pressed();

    /// Preview the position corresponding to the slider drag.
    void on_positionSlider_sliderMoved( double s /**< [in] slider percentage moved by the user */ );

    /// Send a move request from the released slider position.
    void on_positionSlider_sliderReleased();

    /// Send a move request from the typed position field.
    void on_position_returnPressed();

    /// Persist an edited step size.
    void on_stepSize_editingFinished();

    /// Send a negative relative move.
    void on_posMinus_pressed();
    /// Send a positive relative move.
    void on_posPlus_pressed();

    /// Increase the step size by one decade.
    void on_posStepMulTen_pressed();
    /// Decrease the step size by one decade.
    void on_posStepDivTen_pressed();

    /// Request a home operation.
    void on_home_pressed();
    /// Request an immediate stop.
    void on_stop_pressed();

    /// End temporary preset-edit mode after the timer expires.
    void setPointEditTimerOut();

  signals:
    /// Start or restart the set-point edit timeout.
    void setPointEditTimerStart( int timeoutMs /**< [in] timeout duration in milliseconds */ );

    /// Queue a GUI refresh onto the widget thread.
    void doUpdateGUI();

  protected:
    /// Update combo-box styling for staged edits.
    virtual void paintEvent( QPaintEvent *e /**< [in] Qt paint event being processed */ );

  private:
    Ui::stage ui;
};

stage::stage( const std::string &stageName, QWidget *Parent, Qt::WindowFlags f )
    : xWidget( Parent, f ), m_stageName{ stageName }
{
    ui.setupUi( this );

    m_winTitle = m_stageName;

    ui.fsmState->device( m_stageName );
    QFont qf = ui.stageName->font();
    qf.setPixelSize( XW_FONT_SIZE + 3 );
    ui.stageName->setFont( qf );

    ui.stageName->setText( m_stageName.c_str() );

    ui.setPoint->setProperty( "isStatus", true );

    ui.position->setAlignment( Qt::AlignCenter );
    ui.stepSize->setAlignment( Qt::AlignCenter );

    m_setPointEditTimer = new QTimer( this );
    connect( m_setPointEditTimer, SIGNAL( timeout() ), this, SLOT( setPointEditTimerOut() ) );
    connect( this, SIGNAL( setPointEditTimerStart( int ) ), m_setPointEditTimer, SLOT( start( int ) ) );
    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ) );

    onDisconnect();
}

stage::~stage()
{
}

void stage::subscribe()
{
    if( !m_parent )
        return;

    m_parent->addSubscriberProperty( this, m_stageName, "fsm" );
    m_parent->addSubscriberProperty( this, m_stageName, "maxpos" );
    m_parent->addSubscriberProperty( this, m_stageName, "position" );
    m_parent->addSubscriberProperty( this, m_stageName, "filter" );
    m_parent->addSubscriberProperty( this, m_stageName, "parked" );
    m_parent->addSubscriberProperty( this, m_stageName, "presetName" );
    m_parent->addSubscriberProperty( this, m_stageName, "filterName" );
    m_parent->addSubscriber( ui.fsmState );

    return;
}

void stage::onConnect()
{
    setWindowTitle( QString( m_winTitle.c_str() ) );

    ui.stepSize->setText( QString::number( m_step ) );
    m_setPointCommandPending = false;
    m_setPointRequested.clear();

    clearFocus();
}

void stage::onDisconnect()
{
    m_appState.clear();
    m_presets.clear();
    m_presetCurrent.clear();
    m_presetTarget.clear();
    m_setPoint.clear();
    m_parked                 = false;
    m_filterWheel            = false;
    m_maxPos                 = 100;
    m_position               = -1e30;
    m_position_changed       = false;
    m_setPointEditing        = STOPPED;
    m_setPointCommandPending = false;
    m_setPointRequested.clear();

    if( m_setPointEditTimer )
    {
        m_setPointEditTimer->stop();
    }

    setWindowTitle( QString( m_winTitle.c_str() ) + QString( " (disconnected)" ) );

    ui.stageName->setEnabled( false );
    ui.fsmState->setEnabled( false );
    ui.setPoint->clear();
    ui.setPoint->setEnabled( false );
    ui.setPointGo->setEnabled( false );
    ui.positionSlider->setEnabled( false );
    ui.position->setText( "---" );
    ui.position->setEnabled( false );
    ui.stepSize->setEnabled( false );

    ui.posMinus->setEnabled( false );
    ui.posPlus->setEnabled( false );
    ui.posStepMulTen->setEnabled( false );
    ui.posStepDivTen->setEnabled( false );

    ui.home->setEnabled( false );
    ui.stop->setEnabled( false );

    ui.fsmState->onDisconnect();
}

void stage::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void stage::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_stageName )
        return;

    if( ipRecv.getName() == "parked" )
    {
        m_parked = false;
        emit doUpdateGUI();
        return;
    }

    if( ipRecv.getName() == "fsm" || ipRecv.getName() == "maxpos" || ipRecv.getName() == "position" ||
        ipRecv.getName() == "filter" || ipRecv.getName() == "presetName" || ipRecv.getName() == "filterName" )
    {
        onDisconnect();
    }
}

void stage::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_stageName )
        return;

    if( ipRecv.getName() == "fsm" )
    {
        if( ipRecv.find( "state" ) )
        {
            m_appState = ipRecv["state"].get<std::string>();
        }
    }

    if( ipRecv.getName() == "maxpos" )
    {
        if( ipRecv.find( "value" ) )
        {
            m_maxPos = ipRecv["value"].get<double>();
        }
    }

    if( ipRecv.getName() == "position" )
    {
        if( ipRecv.find( "current" ) )
        {
            double val = ipRecv["current"].get<double>();
            if( val != m_position )
                m_position_changed = true;
            m_position = val;
        }
    }
    else if( ipRecv.getName() == "parked" )
    {
        if( ipRecv.find( "current" ) )
        {
            m_parked = ( ipRecv["current"].get<int>() != 0 );
        }
    }
    else if( ipRecv.getName() == "filter" )
    {
        m_filterWheel = true;
        if( ipRecv.find( "current" ) )
        {
            double val = ipRecv["current"].get<double>();
            if( val != m_position )
                m_position_changed = true;
            m_position = val;
        }
    }

    if( ipRecv.getName() == "presetName" || ipRecv.getName() == "filterName" )
    {
        if( ipRecv.getName() == "filterName" )
            m_filterWheel = true;

        int         n = 0;
        std::string newName;
        for( auto it = ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it )
        {
            ++n;
            if( ui.setPoint->findText( QString( it->second.getName().c_str() ) ) == -1 )
            {
                ui.setPoint->addItem( QString( it->second.getName().c_str() ) );
            }

            if( it->second.getSwitchState() == pcf::IndiElement::On )
            {
                if( newName != "" )
                {
                    std::cerr << "More than one switch selected in " << ipRecv.getDevice() << "." << ipRecv.getName()
                              << "\n";
                }

                newName    = it->second.getName();
                m_setPoint = newName;

                if( m_setPointCommandPending && newName == m_setPointRequested )
                {
                    if( m_setPointEditTimer )
                    {
                        m_setPointEditTimer->stop();
                    }

                    m_setPointEditing        = STOPPED;
                    m_setPointCommandPending = false;
                    m_setPointRequested.clear();
                }

                if( m_setPointEditing != STARTED && !m_setPointCommandPending )
                {
                    ui.setPoint->setCurrentText( m_setPoint.c_str() );
                }
            }
        }

        if( m_filterWheel )
            m_maxPos = n + 0.5;
    }

    emit doUpdateGUI();
}

void stage::updateGUI()
{
    bool parkedPowerOff = ( m_appState == "POWEROFF" && m_parked );

    if( m_appState != "READY" && m_appState != "OPERATING" && m_appState != "CONFIGURING" && m_appState != "NOTHOMED" &&
        m_appState != "HOMING" && !parkedPowerOff )
    {
        ui.stageName->setEnabled( false );
        ui.fsmState->setEnabled( false );
        ui.setPoint->setEnabled( false );
        ui.setPointGo->setEnabled( false );
        ui.positionSlider->setEnabled( false );
        ui.position->setText( "---" );
        ui.position->setEnabled( false );
        ui.stepSize->setEnabled( false );

        ui.posMinus->setEnabled( false );
        ui.posPlus->setEnabled( false );
        ui.posStepMulTen->setEnabled( false );
        ui.posStepDivTen->setEnabled( false );

        ui.home->setEnabled( false );
        ui.stop->setEnabled( false );

        return;
    }

    if( parkedPowerOff )
    {
        ui.stageName->setEnabled( true );
        ui.fsmState->setEnabled( true );
        ui.setPoint->setEnabled( false );
        ui.setPointGo->setEnabled( false );
        ui.positionSlider->setEnabled( false );
        ui.position->setEnabled( true );
        ui.stepSize->setEnabled( false );

        ui.posMinus->setEnabled( false );
        ui.posPlus->setEnabled( false );
        ui.posStepMulTen->setEnabled( false );
        ui.posStepDivTen->setEnabled( false );

        ui.home->setEnabled( false );
        ui.stop->setEnabled( false );
    }
    else if( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "HOMING" ||
             m_appState == "CONFIGURING" )
    {
        ui.stageName->setEnabled( true );
        ui.fsmState->setEnabled( true );
        ui.setPoint->setEnabled( true );
        ui.stepSize->setEnabled( true );
        ui.posStepMulTen->setEnabled( true );
        ui.posStepDivTen->setEnabled( true );
        ui.stop->setEnabled( true );

        if( m_appState == "READY" )
        {
            ui.setPointGo->setEnabled( true );
            ui.positionSlider->setEnabled( true );
            ui.position->setEnabled( true );
            ui.posMinus->setEnabled( true );
            ui.posPlus->setEnabled( true );
            ui.home->setEnabled( true );
        }
        else
        {
            ui.setPointGo->setEnabled( false );
            ui.positionSlider->setEnabled( false );
            ui.position->setEnabled( false );
            ui.posMinus->setEnabled( false );
            ui.posPlus->setEnabled( false );
            ui.home->setEnabled( false );
        }
    }
    else if( m_appState == "NOTHOMED" )
    {
        ui.stageName->setEnabled( true );
        ui.fsmState->setEnabled( true );
        ui.setPoint->setEnabled( false );
        ui.stepSize->setEnabled( false );
        ui.posStepMulTen->setEnabled( false );
        ui.posStepDivTen->setEnabled( false );
        ui.stop->setEnabled( false );

        ui.setPointGo->setEnabled( false );
        ui.positionSlider->setEnabled( false );
        ui.position->setEnabled( false );
        ui.posMinus->setEnabled( false );
        ui.posPlus->setEnabled( false );
        ui.home->setEnabled( true );
    }

    if( m_position_changed )
    {
        ui.position->setTextChanged( QString::number( m_position ) );
        m_position_changed = false;
    }
    else
    {
        ui.position->setText( QString::number( m_position ) );
    }

    ui.positionSlider->setValue( m_position / m_maxPos * 100. );
    // ui.position->updateGUI();

} // updateGUI()

void stage::clear_focus()
{
}

void stage::on_setPoint_activated( int index )
{
    static_cast<void>( index );

    m_setPointEditing        = STARTED;
    m_setPointCommandPending = false;
    m_setPointRequested.clear();
    emit setPointEditTimerStart( 10000 );
    update();
}

void stage::on_setPointGo_pressed()
{
    std::string selection = ui.setPoint->currentText().toStdString();

    if( selection == "" )
    {
        return;
    }

    try
    {
        pcf::IndiProperty ipSend( pcf::IndiProperty::Switch );
        ipSend.setDevice( m_stageName );
        if( m_filterWheel )
        {
            ipSend.setName( "filterName" );
        }
        else
        {
            ipSend.setName( "presetName" );
        }
        ipSend.setPerm( pcf::IndiProperty::ReadWrite );
        ipSend.setState( pcf::IndiProperty::Idle );
        ipSend.setRule( pcf::IndiProperty::OneOfMany );

        for( int idx = 0; idx < ui.setPoint->count(); ++idx )
        {
            std::string elName = ui.setPoint->itemText( idx ).toStdString();

            if( elName == selection )
            {
                ipSend.add( pcf::IndiElement( elName, pcf::IndiElement::On ) );
            }
            else
            {
                ipSend.add( pcf::IndiElement( elName, pcf::IndiElement::Off ) );
            }
        }

        sendNewProperty( ipSend );
    }
    catch( ... )
    {
        std::cerr << "exception thrown in stage::on_setPointGo_pressed\n";
    }

    m_setPointEditing        = STARTED;
    m_setPointCommandPending = true;
    m_setPointRequested      = selection;
    emit setPointEditTimerStart( 10000 );
    update();
}

void stage::on_positionSlider_sliderMoved( double s )
{
    double epos = s / 100.0 * m_maxPos;

    ui.position->setEditText( QString::number( epos ) );
}

void stage::on_positionSlider_sliderReleased()
{
    ui.position->stopEditing();

    double s      = ui.positionSlider->value();
    double newPos = s / 100.0 * m_maxPos;

    ui.positionSlider->setValue( m_position / m_maxPos * 100. );

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_stageName );
    if( m_filterWheel )
    {
        ip.setName( "filter" );
    }
    else
    {
        ip.setName( "position" );
    }
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = newPos;

    sendNewProperty( ip );
}

void stage::on_position_returnPressed()
{
    double newPos = ui.position->editText().toDouble();

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_stageName );
    if( m_filterWheel )
    {
        ip.setName( "filter" );
    }
    else
    {
        ip.setName( "position" );
    }
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = newPos;

    sendNewProperty( ip );

    ui.position->clearFocus();
}

void stage::on_stepSize_editingFinished()
{
    m_step = ui.stepSize->text().toDouble();
    ui.stepSize->setText( QString::number( m_step ) );
}

void stage::on_posMinus_pressed()
{
    double newPos = m_position - m_step;

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_stageName );
    if( m_filterWheel )
    {
        ip.setName( "filter" );
    }
    else
    {
        ip.setName( "position" );
    }
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = newPos;

    sendNewProperty( ip );
}

void stage::on_posPlus_pressed()
{
    double newPos = m_position + m_step;

    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_stageName );
    if( m_filterWheel )
    {
        ip.setName( "filter" );
    }
    else
    {
        ip.setName( "position" );
    }
    ip.add( pcf::IndiElement( "target" ) );
    ip["target"] = newPos;

    sendNewProperty( ip );
}

void stage::on_posStepMulTen_pressed()
{
    m_step *= 10.0;
    ui.stepSize->setText( QString::number( m_step ) );
}

void stage::on_posStepDivTen_pressed()
{
    m_step /= 10.0;
    ui.stepSize->setText( QString::number( m_step ) );
}

void stage::on_home_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_stageName );
    ipFreq.setName( "home" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void stage::on_stop_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_stageName );
    ipFreq.setName( "stop" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void stage::setPointEditTimerOut()
{
    m_setPointEditing        = STOPPED;
    m_setPointCommandPending = false;
    m_setPointRequested.clear();
    ui.setPoint->setCurrentText( m_setPoint.c_str() );
    update();
}

void stage::paintEvent( QPaintEvent *e )
{
    if( m_setPointEditing == STARTED )
    {
        ui.setPoint->setProperty( "isStatus", false );
        ui.setPoint->setProperty( "isEditing", true );
        style()->unpolish( ui.setPoint );
    }
    else
    {
        ui.setPoint->setProperty( "isEditing", false );
        ui.setPoint->setProperty( "isStatus", true );
        style()->unpolish( ui.setPoint );
    }

    QWidget::paintEvent( e );
}

} // namespace xqt

#include "moc_stage.cpp"

#endif
