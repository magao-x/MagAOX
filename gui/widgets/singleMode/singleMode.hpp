
#ifndef singleMode_hpp
#define singleMode_hpp

#include <mutex>
#include "ui_singleMode.h"

#include "../xWidgets/xWidget.hpp"
#include "../xWidgets/gainCtrl.hpp"
#include "../xWidgets/statusEntry.hpp"

namespace xqt
{

class singleMode : public xWidget
{
    Q_OBJECT

  protected:
    std::string m_procName;
    std::string m_windowTitle;

    std::string m_loopName;
    std::string m_loopNumber;

    std::string m_appState;

    int m_modeNumber{ 0 };

    QTimer *m_updateTimer{ nullptr };

  public:
    singleMode( std::string &procName, QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    ~singleMode();

    void subscribe();

    void onConnect();

    void onDisconnect();

    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    void sendNewGain( double ng );
    void sendNewMultCoeff( double nm );

    void setEnableDisable( bool tf );

  public slots:
    void updateGUI();

    void on_mode_m_pressed();
    void on_mode_p_pressed();

  signals:

    void doUpdateGUI();

  private:
    Ui::singleMode ui;
};

singleMode::singleMode( std::string &procName, QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    ui.setupUi( this );

    m_procName = procName + "gainctrl";

    m_windowTitle = "Single Mode " + procName;

    setWindowTitle( QString( m_windowTitle.c_str() ) );

    m_procName = procName + "gainctrl";

    ui.gain_ctrl->setup( m_procName, "singleGain", "Gain", -1, -1 );

    ui.mc_ctrl->setup( m_procName, "singleMC", "Mult. Coef.", -1, -1 );
    ui.mc_ctrl->makeMultCoeffCtrl();

    ui.mode_n->setup( m_procName, "singleModeNo", statusEntry::INT, "", "" );
    ui.mode_n->setStretch( 0, 0, 6 );

    m_updateTimer = new QTimer;

    connect( m_updateTimer, SIGNAL( timeout() ), this, SLOT( updateGUI() ) );

    m_updateTimer->start( 250 );

    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ) );

    onDisconnect();
}

singleMode::~singleMode()
{
}

void singleMode::subscribe()
{
    if( !m_parent )
        return;

    m_parent->addSubscriberProperty( this, m_procName, "fsm" );
    m_parent->addSubscriberProperty( this, m_procName, "singleModeNo" );

    m_parent->addSubscriber( ui.gain_ctrl );
    m_parent->addSubscriber( ui.mc_ctrl );
    m_parent->addSubscriber( ui.mode_n );

    return;
}

void singleMode::onConnect()
{
    setWindowTitle( QString( m_procName.c_str() ) );

    ui.gain_ctrl->onConnect();
    ui.mc_ctrl->onConnect();
    ui.mode_n->onConnect();
}

void singleMode::onDisconnect()
{
    std::string tit = m_windowTitle + " (disconnected)";
    setWindowTitle( QString( tit.c_str() ) );

    setEnableDisable( false );

    // xWidget::onDisconnect();
    ui.gain_ctrl->onDisconnect();
    ui.mc_ctrl->onDisconnect();
    ui.mode_n->onDisconnect();
}

void singleMode::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void singleMode::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_procName )
        return;

    if( ipRecv.getName() == "fsm" )
    {
        if( ipRecv.find( "state" ) )
        {
            m_appState = ipRecv["state"].get<std::string>();
        }
    }
    else if( ipRecv.getName() == "singleModeNo" )
    {
        if( ipRecv.find( "current" ) )
        {
            m_modeNumber = ipRecv["current"].get<int>();
        }
    }

    emit doUpdateGUI();
}

void singleMode::setEnableDisable( bool tf )
{

    ui.gain_ctrl->setEnabled( tf );
    ui.mc_ctrl->setEnabled( tf );
    ui.mode_n->setEnabled( tf );
}

void singleMode::updateGUI()
{
    if( m_appState == "READY" || m_appState == "OPERATING" )
    {
        setEnableDisable( true );
    }
    else
    {
        setEnableDisable( false );
    }

} // updateGUI()

void singleMode::on_mode_m_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_procName );
    ip.setName( "singleModeNo" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_modeNumber - 1;

    sendNewProperty( ip );
}

void singleMode::on_mode_p_pressed()
{
    pcf::IndiProperty ip( pcf::IndiProperty::Number );

    ip.setDevice( m_procName );
    ip.setName( "singleModeNo" );
    ip.add( pcf::IndiElement( "target" ) );

    ip["target"] = m_modeNumber + 1;

    sendNewProperty( ip );
}

} // namespace xqt

#include "moc_singleMode.cpp"

#endif
