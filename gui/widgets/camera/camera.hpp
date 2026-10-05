/** \file camera.hpp
 * \brief Widget providing the standard camera control panel for cameraGUI.
 *
 * \author Jared R. Males (jaredmales@gmail.com)
 */

#ifndef camera_hpp
#define camera_hpp

#include <mx/app/appConfigurator.hpp>

#include "ui_camera.h"

#include "xWidgets/xWidget.hpp"
#include "xWidgets/fsmDisplay.hpp"
#include "xWidgets/statusEntry.hpp"
#include "xWidgets/statusDisplay.hpp"
#include "xWidgets/stageStatus.hpp"
#include "xWidgets/toggleSlider.hpp"

#include "roi/roi.hpp"

#include "camera/roiStatus.hpp"
#include "camera/shutterStatus.hpp"

#include "stage/stage.hpp"

#include <QThread>

namespace xqt
{

/// A GUI for an XWC Standard Camera
class camera : public xWidget
{
    Q_OBJECT

  protected:
    std::string m_appState; ///< Current FSM state string reported by the camera app.

    std::string m_camName;  ///< INDI device name of the camera.
    std::string m_darkName; ///< INDI device name of the dark-frame helper app.
    std::string m_avgName;  ///< INDI device name of the averaging helper app.

    fsmDisplay *ui_fsmState{ nullptr }; ///< Camera FSM status display.

    statusEntry *ui_tempCCD{ nullptr }; ///< Detector temperature display and optional control.

    statusDisplay *ui_tempStatus{ nullptr }; ///< Detector temperature-control status display.

    QPushButton *ui_reconfigure{ nullptr }; ///< Button requesting a camera reconfiguration.

    std::vector<std::string> m_stageNames; ///< Configured stage widgets associated with this camera.

    std::vector<stageStatus *> ui_stage; ///< Stage status widgets shown in the left column.

    QPushButton *ui_focus{ nullptr }; ///< Optional button requesting the configured goto-focus preset.

    bool m_gotoFocusPresent{ false }; ///< True once the camera exposes the `goto_focus` request property.

    bool m_focusStateKnown{ false }; ///< True once the camera exposes a current `focus.state` value.

    bool m_focusInFocus{ false }; ///< Cached `focus.state` value, true when the camera reports in focus.

    shutterStatus *ui_shutterStatus{ nullptr }; ///< Optional shutter-status widget.

    roiStatus *ui_roiStatus{ nullptr }; ///< Region-of-interest status widget.

    statusCombo *ui_modes{ nullptr }; ///< Camera mode selector/status widget.

    statusCombo *ui_readoutSpd{ nullptr }; ///< Readout-speed selector/status widget.

    statusCombo *ui_vshiftSpd{ nullptr }; ///< Vertical-shift-speed selector/status widget.

    toggleSlider *ui_cropMode{ nullptr }; ///< ROI crop-mode toggle widget.

    statusEntry *ui_expTime{ nullptr }; ///< Exposure-time display and optional control.

    statusEntry *ui_fps{ nullptr }; ///< Frame-rate display and optional control.

    statusEntry *ui_emGain{ nullptr }; ///< EM-gain display and optional control.

    statusEntry *ui_avgTime{ nullptr }; ///< Averaging time control from the averaging helper app.

    toggleSlider *ui_synchro{ nullptr }; ///< Synchronization toggle control.

    QPushButton *ui_takeDarks{ nullptr }; ///< Button requesting a new dark acquisition.

    float m_temp{ -99 }; ///< Cached detector temperature placeholder used during initialization.

    bool m_takingDark{ false }; ///< True while the dark helper reports an active dark acquisition.

    bool m_inUpdate{ false }; ///< Guards against overlapping periodic GUI updates.

    QTimer *m_updateTimer{ nullptr }; ///< Timer for periodic updates

    bool m_connected{ false }; ///< True while the widget is connected to the INDI stream.

  public:
    /// Constructs the standard camera widget.
    explicit camera( std::string    &camName,
                     QWidget        *Parent = 0 /**< [in] parent widget */,
                     Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags */
    );

    /// Destroys the widget.
    ~camera();

    /// Subscribes the widget and any created child widgets to their INDI properties.
    void subscribe();

    /// Handles the initial connection of the widget to its INDI sources.
    virtual void onConnect();

    /// Handles loss of connection to the widget's INDI sources.
    virtual void onDisconnect();

    /// Handles a newly defined INDI property by reusing the set-property path.
    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Handles removal of an INDI property used by this widget.
    void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has been deleted*/ );

    /// Handles updates to any subscribed INDI property used by this widget.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] the property which has changed*/ );

    /// Hides optional child widgets that should not be shown in every configuration.
    void hideAll();

    /// Enables or disables the widget's controls.
    void setEnableDisable( bool tf /**< [in] true to enable the controls, false to disable them */,
                           bool all = true /**< [in] true to include the title and FSM widgets */
    );

    /// Adds the widget's configuration options before the widget is instantiated.
    static void setupConfig( mx::app::appConfigurator &config );

    /// Loads widget configuration from the application configurator.
    void loadConfig( mx::app::appConfigurator &config );

  public slots:

    /// Refreshes the enabled state and child widgets from cached INDI state.
    void updateGUI();

    /// Creates the detector-temperature widget.
    void setup_temp_ccd( bool ro );

    /// Creates the detector temperature-control status widget.
    void setup_tempStatus();

    /// Creates the reconfigure button.
    void setup_reconfigure();

    /// Sends the camera reconfigure request.
    void reconfigure();

    /// Creates one configured stage widget.
    void setup_stage();

    /// Creates the optional goto-focus button.
    void setup_focus();

    /// Creates the shutter-status widget.
    void setup_shutter();

    /// Creates the ROI status widget.
    void setup_roiStatus();

    /// Creates the mode selector/status widget.
    void setup_modes();

    /// Creates the readout-speed selector/status widget.
    void setup_readoutSpd();

    /// Creates the vertical-shift-speed selector/status widget.
    void setup_vshiftSpd();

    /// Creates the ROI crop-mode toggle widget.
    void setup_cropMode();

    /// Creates the exposure-time widget.
    void setup_expTime( bool ro );

    /// Creates the frame-rate widget.
    void setup_fps( bool ro );

    /// Creates the EM-gain widget.
    void setup_emGain( bool ro );

    /// Creates the averaging-time widget.
    void setup_avgTime( bool ro );

    /// Creates the synchronization toggle widget.
    void setup_synchro();

    /// Creates the take-darks button.
    void setup_takeDarks();

    /// Sends the goto-focus request.
    void gotoFocus();

    /// Sends the take-dark request.
    void takeDark();

  signals:

    /// Requests a GUI refresh on the widget thread.
    void doUpdateGUI();

    /// Stops the periodic update timer.
    void updateTimerStop();

    /// Starts the periodic update timer.
    void updateTimerStart( int );

    /// Queues creation of the detector-temperature widget.
    void add_temp_ccd( bool ro );

    /// Queues creation of the temperature-status widget.
    void add_tempStatus();

    /// Queues creation of the reconfigure button.
    void add_reconfigure();

    /// Queues creation of the goto-focus button.
    void add_focus();

    /// Queues creation of the shutter-status widget.
    void add_shutter();

    /// Queues creation of the ROI status widget.
    void add_roiStatus();

    /// Queues creation of the mode selector/status widget.
    void add_modes();

    /// Queues creation of the readout-speed selector/status widget.
    void add_readoutSpd();

    /// Queues creation of the vertical-shift-speed selector/status widget.
    void add_vshiftSpd();

    /// Queues creation of the crop-mode widget.
    void add_cropMode();

    /// Queues creation of the exposure-time widget.
    void add_expTime( bool ro );

    /// Queues creation of the frame-rate widget.
    void add_fps( bool ro );

    /// Queues creation of the EM-gain widget.
    void add_emGain( bool ro );

    /// Queues creation of the averaging-time widget.
    void add_avgTime( bool ro );

    /// Queues creation of the synchronization widget.
    void add_synchro();

    /// Queues creation of the take-darks button.
    void add_takeDarks();

  private:
    /// Resets cached focus-property state when the camera disconnects or removes focus support.
    void clearFocusState();

    /// Returns the base row used for left-column controls beneath the stage widgets.
    int leftColumnBaseRow() const;

    /// Repositions optional left-column controls after stage or focus changes.
    void layoutLeftColumnControls();

    Ui::camera ui;
};

camera::camera( std::string &camName, QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f ), m_camName{ camName }
{
    m_darkName = m_camName + "-dark";
    m_avgName  = m_camName + "-avg";

    ui.setupUi( this );

    m_updateTimer = new QTimer( this );

    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ) );

    connect( m_updateTimer, SIGNAL( timeout() ), this, SLOT( updateGUI() ) );
    connect( this, SIGNAL( updateTimerStop() ), m_updateTimer, SLOT( stop() ) );
    connect( this, SIGNAL( updateTimerStart( int ) ), m_updateTimer, SLOT( start( int ) ) );

    connect( this, SIGNAL( add_temp_ccd( bool ) ), this, SLOT( setup_temp_ccd( bool ) ) );
    connect( this, SIGNAL( add_tempStatus() ), this, SLOT( setup_tempStatus() ) );

    connect( this, SIGNAL( add_reconfigure() ), this, SLOT( setup_reconfigure() ) );

    connect( this, SIGNAL( add_focus() ), this, SLOT( setup_focus() ) );
    connect( this, SIGNAL( add_shutter() ), this, SLOT( setup_shutter() ) );

    connect( this, SIGNAL( add_roiStatus() ), this, SLOT( setup_roiStatus() ) );
    connect( this, SIGNAL( add_modes() ), this, SLOT( setup_modes() ) );

    connect( this, SIGNAL( add_readoutSpd() ), this, SLOT( setup_readoutSpd() ) );
    connect( this, SIGNAL( add_vshiftSpd() ), this, SLOT( setup_vshiftSpd() ) );
    connect( this, SIGNAL( add_cropMode() ), this, SLOT( setup_cropMode() ) );

    connect( this, SIGNAL( add_expTime( bool ) ), this, SLOT( setup_expTime( bool ) ) );
    connect( this, SIGNAL( add_fps( bool ) ), this, SLOT( setup_fps( bool ) ) );
    connect( this, SIGNAL( add_emGain( bool ) ), this, SLOT( setup_emGain( bool ) ) );
    connect( this, SIGNAL( add_avgTime( bool ) ), this, SLOT( setup_avgTime( bool ) ) );
    connect( this, SIGNAL( add_synchro() ), this, SLOT( setup_synchro() ) );

    connect( this, SIGNAL( add_takeDarks() ), this, SLOT( setup_takeDarks() ) );

    QSpacerItem *holder = new QSpacerItem( 10, 0, QSizePolicy::Expanding, QSizePolicy::Expanding );
    ui.grid->addItem( holder, 2, 1, 1, 1 );

    ui_fsmState = new xqt::fsmDisplay( this );
    ui_fsmState->setObjectName( QString::fromUtf8( "fsmState" ) );
    ui.grid->addWidget( ui_fsmState, 1, 0, 1, 1 );
    ui_fsmState->device( m_camName );

    QFont qf = ui.lab_camName->font();
    qf.setPixelSize( XW_FONT_SIZE + 3 );
    ui.lab_camName->setFont( qf );

    ui.lab_camName->setText( m_camName.c_str() );

    onDisconnect();
}

camera::~camera()
{
}

void camera::subscribe()
{
    if( !m_parent )
        return;

    // The empty-name subscription requests the current device property list,
    // but live updates still require exact property subscriptions.
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_camName, "" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_camName, "fsm" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_camName, "focus" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_camName, "goto_focus" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_darkName, "" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_darkName, "start" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_avgName, "" );
    m_parent->addSubscriberProperty( (multiIndiSubscriber *)this, m_avgName, "avgTime" );

    m_parent->addSubscriber( ui_fsmState );

    if( ui_tempCCD )
        m_parent->addSubscriber( ui_tempCCD );
    if( ui_tempStatus )
        m_parent->addSubscriber( ui_tempStatus );

    if( ui_stage.size() > 0 )
    {
        for( size_t n = 0; n < ui_stage.size(); ++n )
        {
            m_parent->addSubscriber( ui_stage[n] );
        }
    }

    if( ui_shutterStatus )
        m_parent->addSubscriber( ui_shutterStatus );
    if( ui_roiStatus )
        m_parent->addSubscriber( ui_roiStatus );
    if( ui_modes )
        m_parent->addSubscriber( ui_modes );
    if( ui_readoutSpd )
        m_parent->addSubscriber( ui_readoutSpd );
    if( ui_vshiftSpd )
        m_parent->addSubscriber( ui_vshiftSpd );
    if( ui_cropMode )
        m_parent->addSubscriber( ui_cropMode );
    if( ui_expTime )
        m_parent->addSubscriber( ui_expTime );
    if( ui_fps )
        m_parent->addSubscriber( ui_fps );
    if( ui_emGain )
        m_parent->addSubscriber( ui_emGain );
    if( ui_avgTime )
        m_parent->addSubscriber( ui_avgTime );
    if( ui_synchro )
        m_parent->addSubscriber( ui_synchro );

    return;
}

void camera::onConnect()
{
    ui.lab_camName->setEnabled( true );

    setWindowTitle( QString( ( m_camName + "Ctrl" ).c_str() ) );

    ui_fsmState->onConnect();

    if( ui_tempCCD )
        ui_tempCCD->onConnect();
    if( ui_tempStatus )
        ui_tempStatus->onConnect();

    if( ui_stage.size() > 0 )
    {
        for( size_t n = 0; n < ui_stage.size(); ++n )
        {
            ui_stage[n]->onConnect();
        }
    }

    if( ui_shutterStatus )
        ui_shutterStatus->onConnect();

    if( ui_roiStatus )
        ui_roiStatus->onConnect();
    if( ui_modes )
        ui_modes->onConnect();
    if( ui_readoutSpd )
        ui_readoutSpd->onConnect();
    if( ui_vshiftSpd )
        ui_vshiftSpd->onConnect();
    if( ui_cropMode )
        ui_cropMode->onConnect();

    if( ui_expTime )
        ui_expTime->onConnect();
    if( ui_fps )
        ui_fps->onConnect();
    if( ui_emGain )
        ui_emGain->onConnect();
    if( ui_avgTime )
        ui_avgTime->onConnect();

    if( ui_synchro )
        ui_synchro->onConnect();

    clearFocus();
    clearFocusState();

    m_connected = true;

    emit doUpdateGUI();
}

void camera::onDisconnect()
{

    setWindowTitle( QString( ( m_camName + "Ctrl" ).c_str() ) + QString( " (disconnected)" ) );

    ui_fsmState->onDisconnect();

    if( ui_tempCCD )
        ui_tempCCD->onDisconnect();
    if( ui_tempStatus )
        ui_tempStatus->onDisconnect();

    if( ui_stage.size() > 0 )
    {
        for( size_t n = 0; n < ui_stage.size(); ++n )
        {
            ui_stage[n]->onDisconnect();
        }
    }

    if( ui_shutterStatus )
        ui_shutterStatus->onDisconnect();

    if( ui_roiStatus )
        ui_roiStatus->onDisconnect();
    if( ui_modes )
        ui_modes->onDisconnect();
    if( ui_readoutSpd )
        ui_readoutSpd->onDisconnect();
    if( ui_vshiftSpd )
        ui_vshiftSpd->onDisconnect();
    if( ui_cropMode )
        ui_cropMode->onDisconnect();

    if( ui_expTime )
        ui_expTime->onDisconnect();
    if( ui_fps )
        ui_fps->onDisconnect();
    if( ui_emGain )
        ui_emGain->onDisconnect();
    if( ui_avgTime )
        ui_avgTime->onDisconnect();

    if( ui_synchro )
        ui_synchro->onDisconnect();

    clearFocus();
    clearFocusState();

    m_connected = false;
    while( m_inUpdate )
    {
        QThread::msleep( 10 ); // Wait to get out of update
    }
    emit updateTimerStop();

    setEnableDisable( false );
}

void camera::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void camera::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_camName && ipRecv.getDevice() != m_darkName && ipRecv.getDevice() != m_avgName )
        return;

    if( ipRecv.getDevice() == m_camName )
    {
        if( ipRecv.getName() == "fsm" )
        {
            m_appState.clear();
        }
        else if( ipRecv.getName() == "focus" )
        {
            m_focusStateKnown = false;
            m_focusInFocus    = false;
        }
        else if( ipRecv.getName() == "goto_focus" )
        {
            m_gotoFocusPresent = false;
            layoutLeftColumnControls();
        }
    }
    else if( ipRecv.getDevice() == m_darkName )
    {
        if( ipRecv.getName() == "start" )
        {
            m_takingDark = false;
        }
    }

    emit doUpdateGUI();
}

void camera::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_camName && ipRecv.getDevice() != m_darkName && ipRecv.getDevice() != m_avgName )
        return;

    if( ipRecv.getDevice() == m_camName )
    {
        if( ipRecv.getName() == "fsm" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_appState = ipRecv["state"].get<std::string>();
            }
        }

        if( ipRecv.getName() == "temp_ccd" )
        {
            if( !ui_tempCCD )
            {
                bool ro = true;
                if( ipRecv.find( "target" ) )
                    ro = false;

                emit add_temp_ccd( ro );
            }
        }

        if( ipRecv.getName() == "temp_control" )
        {
            if( !ui_tempStatus )
            {
                emit add_tempStatus();
            }
        }

        if( ipRecv.getName() == "reconfigure" )
        {
            if( !ui_reconfigure )
            {
                emit add_reconfigure();
            }
        }

        if( ipRecv.getName() == "focus" )
        {
            if( ipRecv.find( "state" ) )
            {
                m_focusStateKnown = true;
                m_focusInFocus    = ( ipRecv["state"].getSwitchState() == pcf::IndiElement::On );
            }
        }

        if( ipRecv.getName() == "goto_focus" )
        {
            m_gotoFocusPresent = true;

            if( !ui_focus )
            {
                emit add_focus();
            }
            else
            {
                layoutLeftColumnControls();
            }
        }

        if( ipRecv.getName() == "shutter" )
        {
            if( !ui_shutterStatus )
            {
                emit add_shutter();
            }
        }

        if( ipRecv.getName() == "roi_set" )
        {
            if( !ui_roiStatus )
            {
                emit add_roiStatus();
            }
        }

        if( ipRecv.getName() == "mode" )
        {
            if( !ui_modes )
            {
                emit add_modes();
            }
        }

        if( ipRecv.getName() == "readout_speed" )
        {
            if( !ui_readoutSpd )
            {
                emit add_readoutSpd();
            }
        }

        if( ipRecv.getName() == "vshift_speed" )
        {
            if( !ui_vshiftSpd )
            {
                emit add_vshiftSpd();
            }
        }

        if( ipRecv.getName() == "roi_crop_mode" )
        {
            if( !ui_cropMode )
            {
                emit add_cropMode();
            }
        }

        if( ipRecv.getName() == "exptime" )
        {
            if( !ui_expTime )
            {
                bool ro = true;
                if( ipRecv.find( "target" ) )
                    ro = false;

                emit add_expTime( ro );
            }
        }

        if( ipRecv.getName() == "fps" )
        {
            if( !ui_fps )
            {
                bool ro = true;
                if( ipRecv.find( "target" ) )
                    ro = false;

                emit add_fps( ro );
            }
        }

        if( ipRecv.getName() == "emgain" )
        {
            if( !ui_emGain )
            {
                bool ro = true;
                if( ipRecv.find( "target" ) )
                    ro = false;

                emit add_emGain( ro );
            }
        }

        if( ipRecv.getName() == "synchro" )
        {
            if( !ui_synchro )
            {
                emit add_synchro();
            }
        }
    }
    else if( ipRecv.getDevice() == m_darkName )
    {
        if( !ui_takeDarks )
            emit add_takeDarks();

        if( ipRecv.getName() == "start" && ipRecv.find( "toggle" ) )
        {
            if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
            {
                m_takingDark = true;
            }
            else
            {
                m_takingDark = false;
            }
        }
    }
    else if( ipRecv.getDevice() == m_avgName )
    {
        if( !ui_avgTime )
        {
            emit add_avgTime( false );
        }
    }

    emit doUpdateGUI();
}

void camera::hideAll()
{
    if( ui_roiStatus )
        ui_roiStatus->hide();
}

void camera::setEnableDisable( bool tf, bool all )
{
    if( all )
    {
        ui.lab_camName->setEnabled( tf );
        ui_fsmState->setEnabled( tf );
    }

    if( ui_reconfigure )
        ui_reconfigure->setEnabled( tf );
    if( ui_tempCCD )
        ui_tempCCD->setEnabled( tf );
    if( ui_tempStatus )
        ui_tempStatus->setEnabled( tf );
    if( ui_focus )
        ui_focus->setEnabled( tf );

    if( ui_roiStatus )
        ui_roiStatus->setEnabled( tf );
    if( ui_modes )
        ui_modes->setEnabled( tf );
    if( ui_readoutSpd )
        ui_readoutSpd->setEnabled( tf );
    if( ui_vshiftSpd )
        ui_vshiftSpd->setEnabled( tf );
    if( ui_expTime )
        ui_expTime->setEnabled( tf );
    if( ui_fps )
        ui_fps->setEnabled( tf );
    if( ui_emGain )
        ui_emGain->setEnabled( tf );
    if( ui_avgTime )
        ui_avgTime->setEnabled( tf );

    if( ui_stage.size() > 0 )
    {
        for( size_t n = 0; n < ui_stage.size(); ++n )
        {
            ui_stage[n]->setEnabled( tf );
        }
    }

    if( ui_shutterStatus )
        ui_shutterStatus->setEnabled( tf );

    if( ui_takeDarks )
        ui_takeDarks->setEnabled( tf );
}

void camera::setupConfig( mx::app::appConfigurator &config )
{
    config.add( "camera.stages",
                "",
                "camera.stages",
                mx::app::argType::Required,
                "camera",
                "stages",
                false,
                "vector<string>",
                "List of stages associated with this camera" );
}

void camera::loadConfig( mx::app::appConfigurator &config )
{
    config( m_stageNames, "camera.stages" );
    for( size_t n = 0; n < m_stageNames.size(); ++n )
    {
        setup_stage();
    }
    onDisconnect();
}

void camera::updateGUI()
{
    if( m_inUpdate || !m_connected )
        return;
    emit updateTimerStop();
    m_inUpdate = true;

    if( m_appState == "NODEVICE" || m_appState == "NOTCONNECTED" || m_appState == "CONNECTED" )
    {
        setEnableDisable( false, false );
        ui.lab_camName->setEnabled( true );
        ui_fsmState->setEnabled( true );
    }
    else if( m_appState != "READY" && m_appState != "OPERATING" && m_appState != "CONFIGURING" )
    {
        setEnableDisable( false );
        m_inUpdate = false;

        emit updateTimerStart( 1000 );
        return;
    }
    else // if( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "CONFIGURING")
    {
        setEnableDisable( true );
    }

    // Update the component GUIs to ensure they update for connection state, etc.
    if( ui_tempCCD )
        ui_tempCCD->updateGUI();
    if( ui_roiStatus )
        ui_roiStatus->updateGUI();

    if( ui_stage.size() > 0 )
    {
        for( size_t n = 0; n < ui_stage.size(); ++n )
        {
            ui_stage[n]->updateGUI();
        }
    }

    if( ui_shutterStatus )
        ui_shutterStatus->updateGUI();
    if( ui_modes )
        ui_modes->updateGUI();
    if( ui_readoutSpd )
        ui_readoutSpd->updateGUI();
    if( ui_vshiftSpd )
        ui_vshiftSpd->updateGUI();
    if( ui_cropMode )
        ui_cropMode->updateGUI();
    if( ui_expTime )
        ui_expTime->updateGUI();
    if( ui_fps )
        ui_fps->updateGUI();
    if( ui_emGain )
        ui_emGain->updateGUI();
    if( ui_avgTime )
        ui_avgTime->updateGUI();
    if( ui_synchro )
        ui_synchro->updateGUI();

    if( ui_focus )
    {
        ui_focus->setVisible( m_gotoFocusPresent );

        if( ( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "CONFIGURING" ) &&
            m_gotoFocusPresent && m_focusStateKnown && !m_focusInFocus )
        {
            ui_focus->setEnabled( true );
        }
        else
        {
            ui_focus->setEnabled( false );
        }
    }

    if( ( m_appState == "READY" || m_appState == "OPERATING" ) && ui_takeDarks )
    {
        if( m_takingDark )
        {
            ui_takeDarks->setEnabled( false );
        }
        else
        {
            ui_takeDarks->setEnabled( true );
        }
    }

    emit updateTimerStart( 1000 );
    m_inUpdate = false;

} // updateGUI()

void camera::setup_temp_ccd( bool ro )
{
    if( ui_tempCCD )
        return;

    ui_tempCCD = new statusEntry( this );
    ui_tempCCD->setObjectName( QString::fromUtf8( "tempCCD" ) );
    ui_tempCCD->setup( m_camName, "temp_ccd", statusEntry::FLOAT, "Detector Temp.", "C" );
    ui_tempCCD->highlightChanges( false );
    ui_tempCCD->readOnly( ro );

    ui.grid->addWidget( ui_tempCCD, 0, 1, 1, 1 );

    ui_tempCCD->onDisconnect();

    m_parent->addSubscriber( ui_tempCCD );
}

void camera::setup_tempStatus()
{
    if( ui_tempStatus )
        return;

    ui_tempStatus =
        new statusDisplay( m_camName, "temp_control", "status", "Temp. Ctrl.", "", this, Qt::WindowFlags() );
    ui_tempStatus->setObjectName( QString::fromUtf8( "tempStatus" ) );

    ui.grid->addWidget( ui_tempStatus, 1, 1, 1, 1 );

    ui_tempStatus->onDisconnect();

    m_parent->addSubscriber( ui_tempStatus );
}

void camera::setup_reconfigure()
{
    if( ui_reconfigure )
        return;

    ui_reconfigure = new QPushButton( this );
    ui_reconfigure->setObjectName( QString::fromUtf8( "reconfigure" ) );
    ui_reconfigure->setText( "reconfigure" );
    ui_reconfigure->setMaximumWidth( 200 );
    connect( ui_reconfigure, SIGNAL( pressed() ), this, SLOT( reconfigure() ) );
    ui.grid->addWidget( ui_reconfigure, 3, 0, 1, 1, Qt::AlignHCenter );
}

void camera::reconfigure()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_camName );
    ipFreq.setName( "reconfigure" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void camera::setup_stage()
{
    if( ui_stage.size() >= m_stageNames.size() )
        return;

    size_t n = ui_stage.size();

    ui_stage.push_back( new stageStatus( m_stageNames[n], this ) );

    ui_stage[n]->setObjectName( QString::fromUtf8( m_stageNames[n].c_str() ) );

    ui.grid->addWidget( ui_stage[n], 4 + n, 0, 1, 1 );

    ui_stage[n]->onDisconnect();

    layoutLeftColumnControls();
}

void camera::setup_focus()
{
    if( ui_focus )
        return;

    ui_focus = new QPushButton( this );
    ui_focus->setObjectName( QString::fromUtf8( "focus" ) );
    ui_focus->setText( "goto focus" );
    ui_focus->setMaximumWidth( 200 );
    ui_focus->setFocusPolicy( Qt::NoFocus );
    ui_focus->setVisible( m_gotoFocusPresent );
    connect( ui_focus, SIGNAL( pressed() ), this, SLOT( gotoFocus() ) );

    layoutLeftColumnControls();
}

void camera::setup_shutter()
{
    if( ui_shutterStatus )
        return;

    ui_shutterStatus = new shutterStatus( m_camName, this );

    ui_shutterStatus->setObjectName( QString::fromUtf8( "shutter" ) );
    layoutLeftColumnControls();

    ui_shutterStatus->onDisconnect();

    m_parent->addSubscriber( ui_shutterStatus );
}

void camera::setup_roiStatus()
{
    if( ui_roiStatus )
        return; // can get called from several threads

    ui_roiStatus = new roiStatus( m_camName, this );
    ui_roiStatus->setObjectName( QString::fromUtf8( "roiStatus" ) );

    ui.grid->addWidget( ui_roiStatus, 3, 1, 1, 1 );

    ui_roiStatus->onDisconnect();

    m_parent->addSubscriber( ui_roiStatus );
}

void camera::setup_modes()
{
    if( ui_modes )
        return;

    ui_modes = new statusCombo( m_camName, "mode", "", "Mode", "", this );

    ui_modes->setObjectName( QString::fromUtf8( "modes" ) );

    ui.grid->addWidget( ui_modes, 3, 1, 1, 1 );

    ui_modes->onDisconnect();

    m_parent->addSubscriber( ui_modes );
}

void camera::setup_readoutSpd()
{
    if( ui_readoutSpd )
        return;

    ui_readoutSpd = new statusCombo( m_camName, "readout_speed", "", "Readout Spd", "", this );
    ui_readoutSpd->ctrlWidget( nullptr );

    ui_readoutSpd->setObjectName( QString::fromUtf8( "readoutSpd" ) );

    ui.grid->addWidget( ui_readoutSpd, 4, 1, 1, 1 );

    ui_readoutSpd->onDisconnect();

    m_parent->addSubscriber( ui_readoutSpd );
}

void camera::setup_vshiftSpd()
{
    if( ui_vshiftSpd )
        return;

    ui_vshiftSpd = new statusCombo( m_camName, "vshift_speed", "", "Vert. Shift Spd", "", this );
    ui_vshiftSpd->ctrlWidget( nullptr );

    ui_vshiftSpd->setObjectName( QString::fromUtf8( "vshiftSpd" ) );

    ui.grid->addWidget( ui_vshiftSpd, 5, 1, 1, 1 );

    ui_vshiftSpd->onDisconnect();

    m_parent->addSubscriber( ui_vshiftSpd );
}

void camera::setup_cropMode()
{
    if( ui_cropMode )
        return;

    ui_cropMode = new toggleSlider( m_camName, "roi_crop_mode", "Crop Mode", this );
    ui_cropMode->setObjectName( QString::fromUtf8( "cropMode" ) );

    ui.grid->addWidget( ui_cropMode, 6, 1, 1, 1 );

    ui_cropMode->onDisconnect();

    m_parent->addSubscriber( ui_cropMode );
}

void camera::setup_expTime( bool ro )
{
    if( ui_expTime )
        return;

    ui_expTime = new statusEntry( this );
    ui_expTime->setObjectName( QString::fromUtf8( "expTime" ) );
    ui_expTime->setup( m_camName, "exptime", statusEntry::FLOAT, "Exp. Time", "sec" );
    ui_expTime->highlightChanges( true );
    ui_expTime->readOnly( ro );

    ui.grid->addWidget( ui_expTime, 7, 1, 1, 1 );

    ui_expTime->onDisconnect();

    m_parent->addSubscriber( ui_expTime );
}

void camera::setup_fps( bool ro )
{
    if( ui_fps )
        return;

    ui_fps = new statusEntry( this );
    ui_fps->setObjectName( QString::fromUtf8( "fps" ) );
    ui_fps->setup( m_camName, "fps", statusEntry::FLOAT, "Frame Rate", "F.P.S." );
    ui_fps->highlightChanges( true );
    ui_fps->readOnly( ro );

    ui.grid->addWidget( ui_fps, 8, 1, 1, 1 );

    ui_fps->onDisconnect();

    m_parent->addSubscriber( ui_fps );
}

void camera::setup_emGain( bool ro )
{
    if( ui_emGain )
        return;

    ui_emGain = new statusEntry( this );
    ui_emGain->setObjectName( QString::fromUtf8( "emgain" ) );
    ui_emGain->setup( m_camName, "emgain", statusEntry::FLOAT, "E.M. Gain", "" );
    ui_emGain->highlightChanges( true );
    ui_emGain->readOnly( ro );

    ui.grid->addWidget( ui_emGain, 9, 1, 1, 1 );

    ui_emGain->onDisconnect();

    m_parent->addSubscriber( ui_emGain );
}

void camera::setup_avgTime( bool ro )
{
    if( ui_avgTime )
        return;

    ui_avgTime = new statusEntry( this );
    ui_avgTime->setObjectName( QString::fromUtf8( "avgTime" ) );
    ui_avgTime->setup( m_avgName, "avgTime", statusEntry::FLOAT, "Avg. Time", "" );
    ui_avgTime->highlightChanges( true );
    ui_avgTime->readOnly( ro );

    ui.grid->addWidget( ui_avgTime, 11, 1, 1, 1 );

    ui_avgTime->onDisconnect();

    m_parent->addSubscriber( ui_avgTime );
}

void camera::setup_synchro()
{
    if( ui_synchro )
        return;

    ui_synchro = new toggleSlider( m_camName, "synchro", "Synchro", this );
    ui_synchro->setObjectName( QString::fromUtf8( "synchro" ) );

    ui.grid->addWidget( ui_synchro, 10, 1, 1, 1 );

    ui_synchro->onDisconnect();

    m_parent->addSubscriber( ui_synchro );
}

void camera::setup_takeDarks()
{
    if( ui_takeDarks )
        return;

    ui_takeDarks = new QPushButton( this );
    ui_takeDarks->setObjectName( QString::fromUtf8( "takeDarks" ) );
    ui_takeDarks->setText( "take darks" );
    ui_takeDarks->setMaximumWidth( 200 );
    ui_takeDarks->setFocusPolicy( Qt::NoFocus );
    connect( ui_takeDarks, SIGNAL( pressed() ), this, SLOT( takeDark() ) );

    layoutLeftColumnControls();
}

void camera::gotoFocus()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_camName );
    ipFreq.setName( "goto_focus" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void camera::takeDark()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_darkName );
    ipFreq.setName( "start" );
    ipFreq.add( pcf::IndiElement( "toggle" ) );
    ipFreq["toggle"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void camera::clearFocusState()
{
    m_gotoFocusPresent = false;
    m_focusStateKnown  = false;
    m_focusInFocus     = false;

    if( ui_focus )
    {
        ui_focus->hide();
        ui_focus->setEnabled( false );
    }

    layoutLeftColumnControls();
}

int camera::leftColumnBaseRow() const
{
    int doff = 0;
    if( ui_stage.size() > 4 )
    {
        doff = ui_stage.size() - 4;
    }

    return 8 + doff;
}

void camera::layoutLeftColumnControls()
{
    int row = leftColumnBaseRow();

    if( ui_focus )
    {
        ui.grid->removeWidget( ui_focus );
        if( m_gotoFocusPresent )
        {
            ui.grid->addWidget( ui_focus, row, 0, 1, 1, Qt::AlignHCenter );
            ++row;
        }
    }

    if( ui_shutterStatus )
    {
        ui.grid->removeWidget( ui_shutterStatus );
        ui.grid->addWidget( ui_shutterStatus, row, 0, 1, 1 );
        ++row;
    }

    if( ui_takeDarks )
    {
        ui.grid->removeWidget( ui_takeDarks );
        ui.grid->addWidget( ui_takeDarks, row, 0, 1, 1, Qt::AlignHCenter );
    }
}

} // namespace xqt

#include "moc_camera.cpp"

#endif
