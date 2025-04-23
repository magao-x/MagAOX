#ifndef xqt_pwr_hpp
#define xqt_pwr_hpp

#include <mutex>

#include <QStyle>

#include <mx/app/appConfigurator.hpp>

#include "ui_pwr.h"

#include "xWidgets/xWidget.hpp"

#include "pwrDevice.hpp"
#include "pwrChannel.hpp"

namespace xqt
{

class pwr : public xWidget
{
    Q_OBJECT

  protected:
    std::vector<xqt::pwrDevice *> m_devices;      ///< The user devices
    std::vector<xqt::pwrDevice *> m_adminDevices; ///< The admin devices

    /// Mutex for locking INDI communications.
    std::mutex m_addMutex;

  public:
    pwr( const std::string &devName, QWidget *Parent = 0, Qt::WindowFlags f = Qt::WindowFlags() );

    virtual ~pwr() noexcept;

    virtual void subscribe();

    virtual void onConnect();

    /// Called by the parent once the parent is disconnected.
    /** If this is reimplemented, you should call pwr::onDisconnect() to ensure children are notified.
     *
     */
    virtual void onDisconnect();

    /// Callback for a `defProperty` message notifying us that the propery has changed.
    /** This is called by the publisher which is subscribed to.
     *
     */
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been defined*/ );

    /// Callback for a SET PROPERTY message notifying us that the propery has changed.
    /** This is called by the publisher which is subscribed to.
     *
     */
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    // static b/c it gets called before the device is instantiated.
    static void setupConfig( mx::app::appConfigurator &config );

    void loadConfig( mx::app::appConfigurator &config );

    /// Populate the switch grid
    /** Adds each device and its switches
     */
    void populateGrid();

  public slots:

    /// A channel state change has been requested.
    /** Sends the newProperty request.
     */
    void chChange( pcf::IndiProperty &ip /**< [in] the INDI property to send in the newProperty*/ );

    /// Update the gauges when values have changed
    void updateGauges();

    void on_tabWidget_currentChanged(int index)
    {
        if(index == 1)
        {
            ui.tabWidget->setProperty( "isAdmin", true );
        }
        else
        {
            ui.tabWidget->setProperty( "isAdmin", false );
        }
        style()->unpolish(ui.tabWidget);
    }
  private:
    Ui::pwr ui;
};

pwr::pwr( const std::string &devName, QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    static_cast<void>( devName );

    ui.setupUi( this );

    ui.tabWidget->setCurrentIndex( 0 );

    ui.totalCurrent->setProperty( "isStatus", true );
    ui.averageVoltage->setProperty( "isStatus", true );
    ui.averageFrequency->setProperty( "isStatus", true );
ui.tabWidget->setProperty( "isAdmin", false );
    onDisconnect();
}

pwr::~pwr() noexcept
{
    if( m_parent )
    {
        m_parent->unsubscribe( this );
    }
}

void pwr::subscribe()
{
    for( size_t n = 0; n < m_devices.size(); ++n )
    {
        QObject::connect(
            m_devices[n], SIGNAL( chChange( pcf::IndiProperty & ) ), this, SLOT( chChange( pcf::IndiProperty & ) ) );
        QObject::connect( m_devices[n], SIGNAL( loadChanged() ), this, SLOT( updateGauges() ) );

        m_parent->addSubscriberProperty( this, m_devices[n]->deviceName(), "load" );
        m_parent->addSubscriberProperty( this, m_devices[n]->deviceName(), "channelOutlets" );
        m_parent->addSubscriberProperty( this, m_devices[n]->deviceName(), "channelOnDelays" );
        m_parent->addSubscriberProperty( this, m_devices[n]->deviceName(), "channelOffDelays" );

        for( size_t i = 0; i < m_devices[n]->numChannels(); ++i )
        {
            m_parent->addSubscriberProperty(
                this, m_devices[n]->deviceName(), m_devices[n]->channel( i )->channelName() );
        }
    }

    for( size_t n = 0; n < m_adminDevices.size(); ++n )
    {
        QObject::connect( m_adminDevices[n],
                          SIGNAL( chChange( pcf::IndiProperty & ) ),
                          this,
                          SLOT( chChange( pcf::IndiProperty & ) ) );
        QObject::connect( m_adminDevices[n], SIGNAL( loadChanged() ), this, SLOT( updateGauges() ) );

        m_parent->addSubscriberProperty( this, m_adminDevices[n]->deviceName(), "load" );
        m_parent->addSubscriberProperty( this, m_adminDevices[n]->deviceName(), "channelOutlets" );
        m_parent->addSubscriberProperty( this, m_adminDevices[n]->deviceName(), "channelOnDelays" );
        m_parent->addSubscriberProperty( this, m_adminDevices[n]->deviceName(), "channelOffDelays" );

        for( size_t i = 0; i < m_adminDevices[n]->numChannels(); ++i )
        {
            m_parent->addSubscriberProperty(
                this, m_adminDevices[n]->deviceName(), m_adminDevices[n]->channel( i )->channelName() );
        }
    }
}

void pwr::onConnect()
{
    ui.tabWidget->setEnabled( true );
    ui.tabWidget->setVisible( true );

    setWindowTitle( QString( "pwrCtrl" ) );
}

void pwr::onDisconnect()
{
    ui.tabWidget->setEnabled( false );
    ui.tabWidget->setVisible( false );

    ui.totalCurrent->display( "-.-" );
    ui.averageVoltage->display( "---.-" );
    ui.averageFrequency->display( "--.-" );

    setWindowTitle( QString( "pwrCtrl disconnected" ) );

    multiIndiSubscriber::onDisconnect();
}

void pwr::handleDefProperty( const pcf::IndiProperty &ipRecv /* [in] the property which has changed*/ )
{
    handleSetProperty( ipRecv );
}

void pwr::handleSetProperty( const pcf::IndiProperty &ipRecv /* [in] the property which has changed*/ )
{
    for( size_t n = 0; n < m_devices.size(); ++n )
    {
        if( ipRecv.getDevice() == m_devices[n]->deviceName() )
        {
            m_devices[n]->handleSetProperty( ipRecv );

            break;
        }
    }

    for( size_t n = 0; n < m_adminDevices.size(); ++n )
    {
        if( ipRecv.getDevice() == m_adminDevices[n]->deviceName() )
        {
            m_adminDevices[n]->handleSetProperty( ipRecv );

            return;
        }
    }
}

void pwr::populateGrid()
{
    std::sort( m_devices.begin(), m_devices.end(), compPwrDevice );

    int currRow = 0;
    for( size_t n = 0; n < m_devices.size(); ++n )
    {
        ui.switchGrid->addWidget( m_devices[n]->deviceNameLabel(), currRow, 0, 2, 1 );
        m_devices[n]->deviceNameLabel()->show();

        for( size_t i = 0; i < m_devices[n]->numChannels(); ++i )
        {
            ui.switchGrid->addWidget( m_devices[n]->channel( i )->channelNameLabel(), currRow, i + 1 );
            m_devices[n]->channel( i )->channelNameLabel()->show();
            ui.switchGrid->addWidget( m_devices[n]->channel( i )->channelSwitch(), currRow + 1, i + 1 );
            m_devices[n]->channel( i )->channelSwitch()->show();
        }

        currRow += 2;
    }

    std::sort( m_adminDevices.begin(), m_adminDevices.end(), compPwrDevice );

    currRow = 0;
    for( size_t n = 0; n < m_adminDevices.size(); ++n )
    {
        ui.switchGridAdmin->addWidget( m_adminDevices[n]->deviceNameLabel(), currRow, 0, 2, 1 );
        m_adminDevices[n]->deviceNameLabel()->show();

        for( size_t i = 0; i < m_adminDevices[n]->numChannels(); ++i )
        {
            ui.switchGridAdmin->addWidget( m_adminDevices[n]->channel( i )->channelNameLabel(), currRow, i + 1 );
            m_adminDevices[n]->channel( i )->channelNameLabel()->show();
            ui.switchGridAdmin->addWidget( m_adminDevices[n]->channel( i )->channelSwitch(), currRow + 1, i + 1 );
             m_adminDevices[n]->channel( i )->channelSwitch()->setProperty("isAdmin", true);
            m_adminDevices[n]->channel( i )->channelSwitch()->show();
        }

        currRow += 2;
    }
}

void pwr::setupConfig( mx::app::appConfigurator &config )
{
    static_cast<void>( config ); // unused
}

void pwr::loadConfig( mx::app::appConfigurator &config )
{
    std::vector<std::string> sections;

    config.unusedSections( sections );

    if( sections.size() == 0 )
    {
        throw std::runtime_error( "no power devices found in config file" );
    }

    for( size_t i = 0; i < sections.size(); ++i )
    {
        std::vector<std::string> admin; // = new std::vector<std::string>;
        std::vector<std::string> user;  // = new std::vector<std::string>;

        config.configUnused( admin, mx::app::iniFile::makeKey( sections[i], "adminChannels" ) );
        config.configUnused( user, mx::app::iniFile::makeKey( sections[i], "userChannels" ) );

        if( user.size() > 0 )
        {
            m_devices.push_back( new pwrDevice( this ) );
            m_devices.back()->deviceName( sections[i] );
            m_devices.back()->setChannels( user );
        }

        if( admin.size() > 0 )
        {
            m_adminDevices.push_back( new pwrDevice( this ) );
            m_adminDevices.back()->deviceName( sections[i] );
            m_adminDevices.back()->setChannels( admin );
        }
    }

    populateGrid();
}

void pwr::chChange( pcf::IndiProperty &ip )
{
    try
    {
        sendNewProperty( ip );
    }
    catch( ... )
    {
        std::cerr << "pwr::chChange Exception caught\n";
    }
}

void pwr::updateGauges()
{
    double sumCurr = 0, sumVolt = 0, sumFreq = 0;
    size_t nCurr = 0, nVolt = 0, nFreq = 0;

    for( size_t i = 0; i < m_devices.size(); ++i )
    {
        double a = m_devices[i]->current();

        if( a >= 0 )
        {
            sumCurr += a;
            ++nCurr;
        }

        double v = m_devices[i]->voltage();
        if( v >= 0 )
        {
            sumVolt += v;
            ++nVolt;
        }

        double f = m_devices[i]->frequency();
        if( f >= 0 )
        {
            sumFreq += f;
            ++nFreq;
        }
    }

    if( nCurr > 0 )
    {
        ui.totalCurrent->display( QString::number( sumCurr, 'f', 1 ) );
    }
    else
    {
        ui.totalCurrent->display( "-.-" );
    }

    if( nVolt > 0 )
    {
        ui.averageVoltage->display( QString::number( sumVolt / nVolt, 'f', 1 ) );
    }
    else
    {
        ui.averageVoltage->display( "---.-" );
    }

    if( nFreq > 0 )
    {
        ui.averageFrequency->display( QString::number( sumFreq / nFreq, 'f', 1 ) );
    }
    else
    {
        ui.averageFrequency->display( "--.-" );
    }
}

} // namespace xqt

#include "moc_pwr.cpp"

#endif // xqt_pwr_hpp
