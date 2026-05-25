/** \file statusCombo.hpp
 * \brief Shared status-combo widget for INDI stateful values.
 */

#ifndef statusCombo_hpp
#define statusCombo_hpp

#include "ui_statusCombo.h"

#include "xWidget.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt
{

/// Combo-box status widget that can fall back to FSM text.
class statusCombo : public xWidget
{
    Q_OBJECT

    enum editchanges
    {
        NOTEDITING,
        STARTED,
        STOPPED
    };

  protected:
    /// Optional detailed control widget opened from this summary row.
    xWidget *m_ctrlWidget{ nullptr };

    /// INDI device name providing this status.
    std::string m_device;
    /// INDI property name used for the selectable value.
    std::string m_property;
    /// Optional INDI element name for direct-value properties.
    std::string m_element;

    /// Label shown to the left of the status field.
    std::string m_label;
    /// Units suffix shown in the label when provided.
    std::string m_units;

    /// Whether value changes should be visually highlighted.
    bool m_highlightChanges{ true };

    /// Whether the displayed value needs repainting.
    bool m_valChanged{ false };

    /// Latest FSM state received for the device.
    std::string m_fsmState;
    /// Latest selected value text received for the device.
    std::string m_value;
    /// Whether the device is explicitly reporting a parked state.
    bool m_parked{ false };
    /// Whether the value should be shown instead of the FSM state.
    bool m_showVal{ true };

    /// Tracks edit state for the combo box.
    int m_statusEditing{ STOPPED };
    /// Whether a sent selection is still waiting for the live state to confirm it.
    bool m_statusCommandPending{ false };
    /// Timer that clears staged combo-box edits after inactivity.
    QTimer *m_statusEditTimer{ nullptr };

  public:
    /// Construct an unconfigured status combo widget.
    statusCombo( QWidget        *Parent /**< [in] owning Qt parent widget */   = 0,
                 Qt::WindowFlags f /**< [in] Qt window flags for the widget */ = Qt::WindowFlags() );

    /// Construct and configure a status combo widget.
    statusCombo( const std::string &device /**< [in] INDI device name displayed by the widget */,
                 const std::string &property /**< [in] INDI property name used for combo-box selections */,
                 const std::string &element /**< [in] optional direct-value element name */,
                 const std::string &label /**< [in] label text shown beside the combo box */,
                 const std::string &units /**< [in] optional units suffix appended to the label */,
                 QWidget           *Parent /**< [in] owning Qt parent widget */   = 0,
                 Qt::WindowFlags    f /**< [in] Qt window flags for the widget */ = Qt::WindowFlags() );

    /// Destroy the widget.
    ~statusCombo();

    /// Configure the device, property, and label metadata for the widget.
    void setup( const std::string &device /**< [in] INDI device name displayed by the widget */,
                const std::string &property /**< [in] INDI property name used for combo-box selections */,
                const std::string &element /**< [in] optional direct-value element name */,
                const std::string &label /**< [in] label text shown beside the combo box */,
                const std::string &units /**< [in] optional units suffix appended to the label */ );

    /// Replace the optional detailed control widget.
    void ctrlWidget( xWidget *cw /**< [in] replacement detailed control widget, or `nullptr` to hide it */ );

    /// Return the optional detailed control widget.
    xWidget *ctrlWidget();

    /// Format the current value text for display.
    virtual QString formatValue();

    /// Subscribe to the required INDI properties.
    virtual void subscribe();

    /// Reset the widget for an active INDI connection.
    virtual void onConnect();

    /// Reset the widget for a disconnected INDI connection.
    virtual void onDisconnect();

    /// Handle a newly defined property using the same path as updates.
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Handle deletion of an INDI property used by this widget.
    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been deleted*/ );

    /// Handle updates to the device FSM, parked state, or value property.
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

  public:
    /// Hide the control widget when this summary row becomes disabled.
    virtual void changeEvent( QEvent *e /**< [in] Qt change event being processed */ );

  public slots:

    /// Refresh the displayed text from cached state.
    virtual void updateGUI();

    /// Mark the combo box as actively edited.
    void on_status_activated( int index /**< [in] combo-box index selected by the user */ );

    /// Send the selected value back to INDI.
    void on_buttonGo_pressed();

    /// Show the detailed control widget, if present.
    void on_buttonCtrl_pressed();

    /// End temporary edit mode after the timer expires.
    void statusEditTimerOut();

  signals:
    /// Start or restart the edit timeout.
    void statusEditTimerStart( int timeoutMs /**< [in] timeout duration in milliseconds */ );

    /// Queue a GUI refresh onto the Qt event loop.
    void doUpdateGUI();

  protected:
    /// Decide whether the widget should show the value instead of FSM text.
    bool shouldShowValue() const;

    /// Update styling that reflects whether the combo box is in edit mode.
    virtual void paintEvent( QPaintEvent *e /**< [in] Qt paint event being processed */ );

    /// Generated Qt UI backing this summary widget.
    Ui::statusCombo ui;
};

statusCombo::statusCombo( QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
}

statusCombo::statusCombo( const std::string &device,
                          const std::string &property,
                          const std::string &element,
                          const std::string &label,
                          const std::string &units,
                          QWidget           *Parent,
                          Qt::WindowFlags    f )
    : xWidget( Parent, f )
{
    setup( device, property, element, label, units );
}

statusCombo::~statusCombo()
{
}

void statusCombo::setup( const std::string &device,
                         const std::string &property,
                         const std::string &element,
                         const std::string &label,
                         const std::string &units )

{
    m_device   = device;
    m_property = property;
    m_element  = element;
    m_label    = label;
    m_units    = units;

    ui.setupUi( this );
    ui.status->setEditable( false );
    ui.status->setProperty( "isStatus", true );

    std::string lab = m_label;
    if( m_units != "" )
    {
        lab += " [" + m_units + "]";
    }

    ui.label->setText( lab.c_str() );

    QFont qf = ui.label->font();
    qf.setPixelSize( XW_FONT_SIZE );
    ui.label->setFont( qf );

    qf = ui.status->font();
    qf.setPixelSize( XW_FONT_SIZE );
    ui.status->setFont( qf );

    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ) );

    m_statusEditTimer = new QTimer( this );

    connect( m_statusEditTimer, SIGNAL( timeout() ), this, SLOT( statusEditTimerOut() ) );

    connect( this, SIGNAL( statusEditTimerStart( int ) ), m_statusEditTimer, SLOT( start( int ) ) );

    onDisconnect();
}

void statusCombo::ctrlWidget( xWidget *cw )
{
    if( m_ctrlWidget )
    {
        m_ctrlWidget->deleteLater();
        m_ctrlWidget = nullptr;
    }

    if( cw == nullptr )
    {
        ui.buttonCtrl->setVisible( false );
    }
    else
    {
        m_ctrlWidget = cw;
        ui.buttonCtrl->setVisible( true );
    }
}

xWidget *statusCombo::ctrlWidget()
{
    return m_ctrlWidget;
}

QString statusCombo::formatValue()
{
    return QString( m_value.c_str() );
}

bool statusCombo::shouldShowValue() const
{
    return ( m_fsmState == "READY" || m_fsmState == "OPERATING" || ( m_fsmState == "POWEROFF" && m_parked ) );
}

void statusCombo::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    m_parent->addSubscriberProperty( this, m_device, "fsm" );
    m_parent->addSubscriberProperty( this, m_device, "parked" );

    if( m_property != "" )
    {
        m_parent->addSubscriberProperty( this, m_device, m_property );
    }

    if( m_ctrlWidget )
    {
        m_parent->addSubscriber( m_ctrlWidget );
    }

    return;
}

void statusCombo::onConnect()
{
    m_valChanged           = true;
    m_statusCommandPending = false;
}

void statusCombo::onDisconnect()
{
    m_fsmState.clear();
    m_value.clear();
    m_parked               = false;
    m_showVal              = false;
    m_valChanged           = false;
    m_statusEditing        = STOPPED;
    m_statusCommandPending = false;

    if( m_statusEditTimer )
    {
        m_statusEditTimer->stop();
    }

    ui.status->clear();
    ui.status->setPlaceholderText( "" );
    ui.status->setCurrentIndex( -1 );
}

void statusCombo::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void statusCombo::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    if( ipRecv.getName() == "parked" )
    {
        if( m_parked )
        {
            m_parked = false;

            bool showVal = shouldShowValue();
            if( showVal != m_showVal )
            {
                m_showVal    = showVal;
                m_valChanged = true;
            }

            emit doUpdateGUI();
        }

        return;
    }

    if( ipRecv.getName() == "fsm" || ipRecv.getName() == m_property )
    {
        onDisconnect();
    }
}

void statusCombo::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    if( ipRecv.getName() == "fsm" )
    {
        if( ipRecv.find( "state" ) )
        {
            std::string fsmState = ipRecv["state"].get();
            bool        showVal  = false;

            if( fsmState != m_fsmState )
            {
                m_valChanged = true;
            }

            m_fsmState = fsmState;
            showVal    = shouldShowValue();

            if( showVal != m_showVal )
            {
                m_valChanged = true;
                m_showVal    = showVal;
            }
        }
    }
    else if( ipRecv.getName() == "parked" )
    {
        if( ipRecv.find( "current" ) )
        {
            bool parked  = ( ipRecv["current"].get<int>() != 0 );
            bool showVal = false;

            if( parked != m_parked )
            {
                m_valChanged = true;
            }

            m_parked = parked;
            showVal  = shouldShowValue();

            if( showVal != m_showVal )
            {
                m_valChanged = true;
                m_showVal    = showVal;
            }
        }
    }
    else if( ipRecv.getName() == m_property )
    {
        if( ipRecv.getType() != pcf::IndiProperty::Switch )
        {
            std::cerr << "statusCombo: property is not a switch\n";
            return;
        }

        std::map<std::string, pcf::IndiElement> elmap = ipRecv.getElements();

        std::string value;

        if( elmap.size() > 0 )
        {
            // Go through all elements in the property, which should be a switch vector
            for( auto it = elmap.begin(); it != elmap.end(); ++it )
            {
                QString name( it->second.getName().c_str() );
                // Check if it's in the list
                if( ui.status->findText( name ) == -1 )
                {
                    if( name != "" && name != "none" )
                    {
                        ui.status->addItem( name );
                    }
                }

                // See if it's on
                if( it->second.getSwitchState() == pcf::IndiElement::On )
                {
                    if( value != "" )
                    {
                        std::cerr << "statusCombo: more than one item selected\n";
                    }

                    value = it->second.getName();
                }
            }

            if( value != m_value )
            {
                m_valChanged = true;
            }
            m_value = value;

            if( m_statusCommandPending && value == ui.status->currentText().toStdString() )
            {
                m_statusEditing        = STOPPED;
                m_statusCommandPending = false;
                m_valChanged           = true;
            }
        }
    }

    emit doUpdateGUI();
}

void statusCombo::changeEvent( QEvent *e )
{
    if( e->type() == QEvent::EnabledChange && !isEnabledTo( nullptr ) )
    {

        if( m_ctrlWidget )
        {
            m_ctrlWidget->hide();
        }
    }
    xWidget::changeEvent( e );
}

void statusCombo::updateGUI()
{
    if( m_statusEditing == STARTED || m_statusCommandPending )
    {
        return;
    }

    if( isEnabled() )
    {
        if( m_showVal )
        {
            if( m_valChanged )
            {
                QString value = formatValue();
                ui.status->setPlaceholderText( value );
                ui.status->setCurrentIndex( -1 );
                m_valChanged = false;
            }
        }
        else
        {
            if( m_valChanged )
            {
                ui.status->setPlaceholderText( m_fsmState.c_str() );
                ui.status->setCurrentIndex( -1 );
                m_valChanged = false;
            }
        }
    }

} // updateGUI()

void statusCombo::on_status_activated( int index )
{
    static_cast<void>( index );

    m_statusEditing        = STARTED;
    m_statusCommandPending = false;
    emit statusEditTimerStart( 10000 );
    update();
}

void statusCombo::on_buttonGo_pressed()
{
    std::string selection = ui.status->currentText().toStdString();

    if( selection == "" )
    {
        return;
    }

    if( m_property == "" )
    {
        return;
    }

    try
    {
        pcf::IndiProperty ipSend( pcf::IndiProperty::Switch );
        ipSend.setDevice( m_device );
        ipSend.setName( m_property );
        ipSend.setPerm( pcf::IndiProperty::ReadWrite );
        ipSend.setState( pcf::IndiProperty::Idle );
        ipSend.setRule( pcf::IndiProperty::OneOfMany );

        for( int idx = 0; idx < ui.status->count(); ++idx )
        {
            std::string elName = ui.status->itemText( idx ).toStdString();

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
        std::cerr << "INDI exception thrown in statusCombo::on_buttonGo_pressed\n";
    }

    m_statusEditing        = STARTED;
    m_statusCommandPending = true;
    emit statusEditTimerStart( 10000 );
    ui.status->clearFocus();
    ui.buttonGo->clearFocus();
    update();
}

void statusCombo::on_buttonCtrl_pressed()
{
    if( m_ctrlWidget )
    {
        m_ctrlWidget->show();
    }
}

void statusCombo::statusEditTimerOut()
{
    m_statusEditing        = STOPPED;
    m_statusCommandPending = false;
    ui.status->setCurrentIndex( -1 );
    ui.status->clearFocus();
    emit doUpdateGUI();
    update();
}

void statusCombo::paintEvent( QPaintEvent *e )
{
    if( m_statusEditing == STARTED )
    {
        ui.status->setProperty( "isStatus", false );
        ui.status->setProperty( "isEditing", true );
        style()->unpolish( ui.status );
    }
    else
    {
        ui.status->setProperty( "isEditing", false );
        ui.status->setProperty( "isStatus", true );
        style()->unpolish( ui.status );
    }

    QWidget::paintEvent( e );
}

} // namespace xqt

#include "moc_statusCombo.cpp"

#endif
