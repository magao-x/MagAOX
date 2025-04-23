#ifndef statusCombo_hpp
#define statusCombo_hpp

#include "ui_statusCombo.h"

#include "xWidget.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt
{

class statusCombo : public xWidget
{
    Q_OBJECT

    enum editchanges{NOTEDITING, STARTED, STOPPED};

protected:
    xWidget *m_ctrlWidget{ nullptr };

    std::string m_device;
    std::string m_property;
    std::string m_element;

    std::string m_label;
    std::string m_units;

    bool m_highlightChanges{ true };

    bool m_valChanged{ false };

    std::string m_fsmState;
    std::string m_value;
    bool m_showVal{ true };

    int m_statusEditing {STOPPED};
    QTimer * m_statusEditTimer {nullptr};

  public:
    statusCombo( QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    statusCombo( const std::string &device,
                 const std::string &property,
                 const std::string &element,
                 const std::string &label,
                 const std::string &units,
                 QWidget *Parent = 0,
                 Qt::WindowFlags f = Qt::WindowFlags() );

    ~statusCombo();

    void setup( const std::string &device,
                const std::string &property,
                const std::string &element,
                const std::string &label,
                const std::string &units );

    void ctrlWidget( xWidget *cw );

    xWidget *ctrlWidget();

    virtual QString formatValue();

    virtual void subscribe();

    virtual void onConnect();

    virtual void onDisconnect();

    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

  public:

    virtual void changeEvent( QEvent *e );

  public slots:

    virtual void updateGUI();

    void on_status_activated( int index );

    void on_buttonGo_pressed();

    void on_buttonCtrl_pressed();

    void statusEditTimerOut();

  signals:

    void statusEditTimerStart(int);

    void doUpdateGUI();

protected:
    virtual void paintEvent(QPaintEvent * e);

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
                          QWidget *Parent,
                          Qt::WindowFlags f ) : xWidget( Parent, f )
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
    m_device = device;
    m_property = property;
    m_element = element;
    m_label = label;
    m_units = units;

    ui.setupUi( this );
    ui.status->setEditable(false);
    ui.status->setProperty("isStatus", true);

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

    m_statusEditTimer = new QTimer(this);

    connect(m_statusEditTimer, SIGNAL(timeout()), this, SLOT(statusEditTimerOut()));

    connect(this, SIGNAL(statusEditTimerStart(int)), m_statusEditTimer, SLOT(start(int)));

    onDisconnect();
}

void statusCombo::ctrlWidget( xWidget *cw )
{
    if( m_ctrlWidget )
    {
        m_ctrlWidget->deleteLater();
    }

    if(cw == nullptr)
    {
        ui.buttonCtrl->setVisible(false);
    }
    else
    {
        m_ctrlWidget = cw;
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

void statusCombo::subscribe()
{
    if( !m_parent )
    {
        return;
    }

    m_parent->addSubscriberProperty( this, m_device, "fsm" );

    if( m_property != "" )
    {
        m_parent->addSubscriberProperty( this, m_device, m_property );
    }

    if( m_ctrlWidget )
    {
        m_ctrlWidget->subscribe();
        m_parent->addSubscriber( m_ctrlWidget );
    }

    return;
}

void statusCombo::onConnect()
{
    m_valChanged = true;
    if( m_ctrlWidget )
    {
        m_ctrlWidget->onConnect();
    }
}

void statusCombo::onDisconnect()
{
    ui.status->setCurrentIndex( -1 );

    if( m_ctrlWidget )
    {
        m_ctrlWidget->onDisconnect();
    }
}

void statusCombo::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
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

            if( fsmState == "READY" || fsmState == "OPERATING" )
            {
                if( fsmState != m_fsmState )
                {
                    m_valChanged = true;
                }

                if( !m_showVal )
                {
                    m_valChanged = true;
                }

                m_showVal = true;
            }
            else
            {
                m_showVal = false;

                if( fsmState != m_fsmState )
                {
                    m_valChanged = true;
                }

                if( m_showVal )
                {
                    m_valChanged = true;
                }
            }

            m_fsmState = fsmState;
        }
    }
    else if( ipRecv.getName() == m_property )
    {
        if(ipRecv.getType() != pcf::IndiProperty::Switch)
        {
            std::cerr << "statusCombo: property is not a switch\n";
            return;
        }

        std::map<std::string, pcf::IndiElement> elmap = ipRecv.getElements();

        std::string value;

        if(elmap.size() > 0 )
        {
            //Go through all elements in the property, which should be a switch vector
            for(auto it = elmap.begin(); it != elmap.end(); ++it)
            {
                QString name(it->second.getName().c_str());
                //Check if it's in the list
                if(ui.status->findText(name) == -1)
                {
                    if(name != "" && name != "none")
                    {
                        ui.status->addItem(name);
                    }
                }

                //See if it's on
                if(it->second.getSwitchState() == pcf::IndiElement::On)
                {
                    if(value != "")
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
    if( isEnabled() )
    {
        if( m_showVal )
        {
            if( m_valChanged )
            {
                QString value = formatValue();
                ui.status->setPlaceholderText(value);
                ui.status->setCurrentIndex(-1);
                m_valChanged = false;
            }
        }
        else
        {
            if( m_valChanged )
            {
                ui.status->setPlaceholderText(m_fsmState.c_str());
                ui.status->setCurrentIndex(-1);
                m_valChanged = false;
            }
        }
    }

} // updateGUI()

void statusCombo::on_status_activated( int index )
{
    static_cast<void>(index);

    m_statusEditing = STARTED;
    emit statusEditTimerStart(10000);
    update();
}

void statusCombo::on_buttonGo_pressed()
{
    std::string selection = ui.status->currentText().toStdString();

    if(selection == "")
    {
        return;
    }

    try
    {
        pcf::IndiProperty ipSend(pcf::IndiProperty::Switch);
        ipSend.setDevice(m_device);
        ipSend.setName(m_property);
        ipSend.setPerm(pcf::IndiProperty::ReadWrite);
        ipSend.setState(pcf::IndiProperty::Idle);
        ipSend.setRule(pcf::IndiProperty::OneOfMany);

        for(int idx = 0; idx < ui.status->count(); ++idx)
        {
            std::string elName = ui.status->itemText(idx).toStdString();

            if(elName == selection)
            {
                ipSend.add(pcf::IndiElement(elName, pcf::IndiElement::On));
            }
            else
            {
                ipSend.add(pcf::IndiElement(elName, pcf::IndiElement::Off));
            }
        }

        sendNewProperty(ipSend);
    }
    catch(...)
    {
        std::cerr << "INDI exception thrown in statusCombo::on_buttonGo_pressed\n";
    }

    m_statusEditing = STOPPED;
    ui.status->setCurrentText(formatValue());
    ui.status->clearFocus();
    ui.buttonGo->clearFocus();
    update();
}

void statusCombo::on_buttonCtrl_pressed()
{
    if( m_ctrlWidget )
    {
        m_ctrlWidget->show();
        m_ctrlWidget->onConnect();
    }
}

void statusCombo::statusEditTimerOut()
{
   m_statusEditing = STOPPED;
   ui.status->setCurrentText(formatValue());
   ui.status->clearFocus();
   update();
}

void statusCombo::paintEvent(QPaintEvent * e)
{
   if(m_statusEditing == STARTED)
   {
      ui.status->setProperty("isStatus", false);
      ui.status->setProperty("isEditing", true);
      style()->unpolish(ui.status);
   }
   else
   {
      ui.status->setProperty("isEditing", false);
      ui.status->setProperty("isStatus", true);
      style()->unpolish(ui.status);
   }

   QWidget::paintEvent(e);
}

} // namespace xqt

#include "moc_statusCombo.cpp"

#endif
