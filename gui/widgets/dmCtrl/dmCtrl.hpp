/** \file dmCtrl.hpp
 * \brief Deformable mirror control widget for INDI-backed operations.
 * \author Jared R. Males
 */

#ifndef dmCtrl_hpp
#define dmCtrl_hpp

#include <QComboBox>
#include <QSignalBlocker>
#include <QTimer>

#include <mutex>
#include <vector>

#include "ui_dmCtrl.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

/// Deformable mirror control widget backed by INDI properties.
class dmCtrl : public xWidget
{
    Q_OBJECT

  protected:
    std::string m_appState; ///< Current FSM state for the DM controller app.

    std::string m_dmName;    ///< INDI device name for the target DM application.
    std::string m_shmimName; ///< Shared-memory stream name reported by the DM app.

    std::string              m_flatShmim;          ///< Shared-memory stream backing the selected flat.
    bool                     m_flatSet{ false };   ///< True when a flat is currently applied.
    std::string              m_flatName;           ///< Name of the selected flat option.
    std::vector<std::string> m_flatOptions;        ///< Available flat options from INDI.
    bool        m_flatPropertyRefreshing{ false }; ///< True while the DM app is redefining the flat options.
    std::string m_flatRequestedName;               ///< Flat option requested by the user but not yet confirmed.
    bool        m_flatSelectionPending{ false };   ///< True while the GUI should hold a requested flat selection.
    QTimer     *m_flatSelectionTimer{ nullptr };   ///< Clears unconfirmed flat selections after a timeout.
    QTimer     *m_flatRefreshTimer{ nullptr };     ///< Clears stale flat options if a redefine never completes.

    std::string              m_testShmim;          ///< Shared-memory stream backing the selected test pattern.
    bool                     m_testSet{ false };   ///< True when a test pattern is currently applied.
    std::string              m_testName;           ///< Name of the selected test option.
    std::vector<std::string> m_testOptions;        ///< Available test options from INDI.
    bool        m_testPropertyRefreshing{ false }; ///< True while the DM app is redefining the test options.
    std::string m_testRequestedName;               ///< Test option requested by the user but not yet confirmed.
    bool        m_testSelectionPending{ false };   ///< True while the GUI should hold a requested test selection.
    QTimer     *m_testSelectionTimer{ nullptr };   ///< Clears unconfirmed test selections after a timeout.
    QTimer     *m_testRefreshTimer{ nullptr };     ///< Clears stale test options if a redefine never completes.

    std::mutex m_stateMutex; ///< Guards cached state copied into the GUI thread.

    bool m_connected{ false }; ///< True once the widget has processed an INDI connect event.
    bool m_inUpdate{ false };  ///< Prevents re-entrant queued GUI refresh work.

  public:
    /// Constructs the DM control widget.
    explicit dmCtrl( std::string    &dmName,                    /**< [in] INDI device name for the DM controller. */
                     QWidget        *Parent = 0,                /**< [in] Optional parent widget. */
                     Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */
    );

    /// Destroys the widget and detaches from parent subscriber if attached.
    ~dmCtrl();

    /// Subscribes this widget to required INDI properties.
    void subscribe();

    /// Handles parent connection events from subscriber framework.
    virtual void onConnect();

    /// Handles parent disconnection events from subscriber framework.
    virtual void onDisconnect();

    /// Handles defProperty notifications.
    void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Property that has changed. */ );

    /// Handles delProperty notifications.
    void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Property that has been deleted. */ );

    /// Handles setProperty notifications.
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Property that has changed. */ );

  public slots:
    /// Applies connected-state widget enablement and display updates.
    void onConnectGUI();

    /// Applies disconnected-state widget enablement and display updates.
    void onDisconnectGUI();

    /// Refreshes GUI controls from cached state.
    void updateGUI();

    /// Sends initDM request.
    void on_buttonInit_pressed();

    /// Sends zeroAll request.
    void on_buttonZeroAll_pressed();

    /// Sends zeroDM request.
    void on_buttonZero_pressed();

    /// Sends releaseDM request.
    void on_buttonRelease_pressed();

    /// Updates selected flat and sends flat selection request.
    void on_comboSelectFlat_activated( int index /**< [in] Combo-box index of selected flat. */ );

    /// Sends request to apply currently selected flat.
    void on_buttonSetFlat_pressed();

    /// Sends request to clear applied flat.
    void on_buttonZeroFlat_pressed();

    /// Updates selected test and sends test selection request.
    void on_comboSelectTest_activated( int index /**< [in] Combo-box index of selected test pattern. */ );

    /// Sends request to apply currently selected test.
    void on_buttonSetTest_pressed();

    /// Sends request to clear applied test.
    void on_buttonZeroTest_pressed();

    /// Clears an unconfirmed flat selection after the timeout expires.
    void flatSelectionTimerOut();

    /// Clears cached flat options after an incomplete property recreation.
    void flatRefreshTimerOut();

    /// Clears an unconfirmed test selection after the timeout expires.
    void testSelectionTimerOut();

    /// Clears cached test options after an incomplete property recreation.
    void testRefreshTimerOut();

  signals:
    /// Queues connected-state GUI updates onto the widget thread.
    void doOnConnect();

    /// Queues disconnected-state GUI updates onto the widget thread.
    void doOnDisconnect();

    /// Queues state refresh updates onto the widget thread.
    void doUpdateGUI();

    /// Stops the pending flat-selection timeout on the widget thread.
    void flatSelectionTimerStop();

    /// Starts the transient flat-refresh timeout on the widget thread.
    void flatRefreshTimerStart( int timeoutMs /**< [in] Timeout duration in milliseconds. */ );

    /// Stops the transient flat-refresh timeout on the widget thread.
    void flatRefreshTimerStop();

    /// Stops the pending test-selection timeout on the widget thread.
    void testSelectionTimerStop();

    /// Starts the transient test-refresh timeout on the widget thread.
    void testRefreshTimerStart( int timeoutMs /**< [in] Timeout duration in milliseconds. */ );

    /// Stops the transient test-refresh timeout on the widget thread.
    void testRefreshTimerStop();

  private:
    /// Replace combo-box contents only when the option list has actually changed.
    void syncComboOptions( QComboBox                      *combo /**< [in] Combo box to synchronize. */,
                           const std::vector<std::string> &options /**< [in] Options that should be displayed. */ );

    Ui::dmCtrl ui; ///< Generated Qt UI object.
};

dmCtrl::dmCtrl( std::string &dmName, QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f ), m_dmName{ dmName }
{
    ui.setupUi( this );

    setWindowTitle( QString( m_dmName.c_str() ) );

    ui.fsmState->NOTHOMED( "RIP" );
    ui.fsmState->HOMING( "INITIALIZING" );
    ui.fsmState->READY( "NOT SET" );
    ui.fsmState->OPERATING( "SET" );
    ui.fsmState->device( m_dmName );

    setXwFont( ui.buttonInit );
    setXwFont( ui.buttonZeroAll );
    setXwFont( ui.buttonZero );
    setXwFont( ui.buttonInit );
    setXwFont( ui.buttonRelease );
    setXwFont( ui.buttonSetFlat );
    setXwFont( ui.buttonZeroFlat );
    setXwFont( ui.buttonSetTest );
    setXwFont( ui.buttonZeroTest );
    setXwFont( ui.comboSelectFlat );
    setXwFont( ui.comboSelectTest );

    setXwFont( ui.labelShmimName );
    setXwFont( ui.labelShmimName_value );
    setXwFont( ui.labelFlatShmim );
    setXwFont( ui.labelFlatShmim_value );
    setXwFont( ui.labelTestShmim );
    setXwFont( ui.labelTestShmim_value );

    connect( this, SIGNAL( doOnConnect() ), this, SLOT( onConnectGUI() ), Qt::QueuedConnection );
    connect( this, SIGNAL( doOnDisconnect() ), this, SLOT( onDisconnectGUI() ), Qt::QueuedConnection );
    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ), Qt::QueuedConnection );

    m_flatSelectionTimer = new QTimer( this );
    m_flatSelectionTimer->setSingleShot( true );
    connect( m_flatSelectionTimer, SIGNAL( timeout() ), this, SLOT( flatSelectionTimerOut() ) );
    connect( this, SIGNAL( flatSelectionTimerStop() ), m_flatSelectionTimer, SLOT( stop() ), Qt::QueuedConnection );

    m_flatRefreshTimer = new QTimer( this );
    m_flatRefreshTimer->setSingleShot( true );
    connect( m_flatRefreshTimer, SIGNAL( timeout() ), this, SLOT( flatRefreshTimerOut() ) );
    connect(
        this, SIGNAL( flatRefreshTimerStart( int ) ), m_flatRefreshTimer, SLOT( start( int ) ), Qt::QueuedConnection );
    connect( this, SIGNAL( flatRefreshTimerStop() ), m_flatRefreshTimer, SLOT( stop() ), Qt::QueuedConnection );

    m_testSelectionTimer = new QTimer( this );
    m_testSelectionTimer->setSingleShot( true );
    connect( m_testSelectionTimer, SIGNAL( timeout() ), this, SLOT( testSelectionTimerOut() ) );
    connect( this, SIGNAL( testSelectionTimerStop() ), m_testSelectionTimer, SLOT( stop() ), Qt::QueuedConnection );

    m_testRefreshTimer = new QTimer( this );
    m_testRefreshTimer->setSingleShot( true );
    connect( m_testRefreshTimer, SIGNAL( timeout() ), this, SLOT( testRefreshTimerOut() ) );
    connect(
        this, SIGNAL( testRefreshTimerStart( int ) ), m_testRefreshTimer, SLOT( start( int ) ), Qt::QueuedConnection );
    connect( this, SIGNAL( testRefreshTimerStop() ), m_testRefreshTimer, SLOT( stop() ), Qt::QueuedConnection );

    onDisconnectGUI();
}

dmCtrl::~dmCtrl()
{
    if( m_parent )
        m_parent->unsubscribe( this );
}

void dmCtrl::subscribe()
{
    if( !m_parent )
        return;

    m_parent->addSubscriberProperty( this, m_dmName, "fsm" );
    m_parent->addSubscriberProperty( this, m_dmName, "sm_shmimName" );
    m_parent->addSubscriberProperty( this, m_dmName, "flat" );
    m_parent->addSubscriberProperty( this, m_dmName, "flat_shmim" );
    m_parent->addSubscriberProperty( this, m_dmName, "flat_set" );
    m_parent->addSubscriberProperty( this, m_dmName, "test" );
    m_parent->addSubscriberProperty( this, m_dmName, "test_shmim" );
    m_parent->addSubscriberProperty( this, m_dmName, "test_set" );
    return;
}

void dmCtrl::onConnect()
{
    emit doOnConnect();
}

void dmCtrl::onConnectGUI()
{
    m_connected = true;

    ui.fsmState->setEnabled( true );
    ui.labelShmimName->setEnabled( true );
    ui.labelShmimName_value->setEnabled( true );
    ui.labelFlatShmim->setEnabled( true );
    ui.labelFlatShmim_value->setEnabled( true );
    ui.labelTestShmim->setEnabled( true );
    ui.labelTestShmim_value->setEnabled( true );

    ui.buttonZeroAll->setEnabled( true );

    ui.comboSelectFlat->setEnabled( true );
    ui.comboSelectTest->setEnabled( true );

    ui.fsmState->onConnect();

    setWindowTitle( QString( m_dmName.c_str() ) );

    emit doUpdateGUI();
}

void dmCtrl::onDisconnect()
{
    emit doOnDisconnect();
}

void dmCtrl::onDisconnectGUI()
{
    m_connected = false;

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        m_appState.clear();
        m_shmimName.clear();
        m_flatShmim.clear();
        m_flatSet = false;
        m_flatName.clear();
        m_flatOptions.clear();
        m_flatPropertyRefreshing = false;
        m_flatRequestedName.clear();
        m_flatSelectionPending = false;
        m_testShmim.clear();
        m_testSet = false;
        m_testName.clear();
        m_testOptions.clear();
        m_testPropertyRefreshing = false;
        m_testRequestedName.clear();
        m_testSelectionPending = false;
    }

    if( m_flatSelectionTimer )
    {
        m_flatSelectionTimer->stop();
    }

    if( m_testSelectionTimer )
    {
        m_testSelectionTimer->stop();
    }

    if( m_flatRefreshTimer )
    {
        m_flatRefreshTimer->stop();
    }

    if( m_testRefreshTimer )
    {
        m_testRefreshTimer->stop();
    }

    ui.fsmState->setEnabled( false );
    ui.labelShmimName->setEnabled( false );
    ui.labelShmimName_value->setEnabled( false );
    ui.labelFlatShmim->setEnabled( false );
    ui.labelFlatShmim_value->setEnabled( false );
    ui.labelTestShmim->setEnabled( false );
    ui.labelTestShmim_value->setEnabled( false );

    ui.buttonInit->setEnabled( false );
    ui.buttonZero->setEnabled( false );
    ui.buttonZeroAll->setEnabled( false );
    ui.buttonRelease->setEnabled( false );

    ui.buttonSetFlat->setEnabled( false );
    ui.buttonZeroFlat->setEnabled( false );

    ui.buttonSetTest->setEnabled( false );
    ui.buttonZeroTest->setEnabled( false );

    ui.comboSelectFlat->setEnabled( false );
    ui.comboSelectTest->setEnabled( false );

    ui.labelShmimName_value->setText( "" );
    ui.labelFlatShmim_value->setText( "" );
    ui.labelTestShmim_value->setText( "" );
    ui.comboSelectFlat->clear();
    ui.comboSelectTest->clear();

    setWindowTitle( QString( m_dmName.c_str() ) + QString( " (disconnected)" ) );

    ui.fsmState->onDisconnect();
}

void dmCtrl::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void dmCtrl::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_dmName )
    {
        return;
    }

    if( ipRecv.getName() == "fsm" )
    {
        emit doOnDisconnect();
        return;
    }

    if( ipRecv.getName() == "sm_shmimName" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_shmimName.clear();
        }

        emit doUpdateGUI();
        return;
    }

    if( ipRecv.getName() == "flat" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_flatPropertyRefreshing = true;
        }

        emit flatRefreshTimerStart( 2000 );

        return;
    }

    if( ipRecv.getName() == "flat_shmim" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_flatShmim.clear();
        }

        emit doUpdateGUI();
        return;
    }

    if( ipRecv.getName() == "flat_set" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_flatSet = false;
        }

        emit doUpdateGUI();
        return;
    }

    if( ipRecv.getName() == "test" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_testPropertyRefreshing = true;
        }

        emit testRefreshTimerStart( 2000 );

        return;
    }

    if( ipRecv.getName() == "test_shmim" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_testShmim.clear();
        }

        emit doUpdateGUI();
        return;
    }

    if( ipRecv.getName() == "test_set" )
    {
        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_testSet = false;
        }

        emit doUpdateGUI();
        return;
    }
}

void dmCtrl::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_dmName )
    {
        return;
    }
    else if( ipRecv.getName() == "fsm" )
    {
        if( ipRecv.find( "state" ) )
        {
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_appState = ipRecv["state"].get<std::string>();
        }

        ui.fsmState->handleSetProperty( ipRecv );
    }
    else if( ipRecv.getName() == "sm_shmimName" )
    {
        if( ipRecv.find( "name" ) )
        {
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_shmimName = ipRecv["name"].get<std::string>();
        }
    }
    else if( ipRecv.getName() == "flat" )
    {
        bool flatStopSelectionTimer = false;
        bool flatRequestedAvailable = false;
        bool flatStopRefreshTimer   = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_flatPropertyRefreshing = false;
            m_flatOptions.clear();
            m_flatName = "";
            for( auto it = ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it )
            {
                m_flatOptions.push_back( it->first );
                if( it->first == m_flatRequestedName )
                {
                    flatRequestedAvailable = true;
                }
                if( ipRecv[it->first] == pcf::IndiElement::On )
                    m_flatName = it->first;
            }

            if( m_flatSelectionPending && m_flatName == m_flatRequestedName )
            {
                m_flatRequestedName.clear();
                m_flatSelectionPending = false;
                flatStopSelectionTimer = true;
            }
            else if( m_flatSelectionPending && !m_flatRequestedName.empty() && !flatRequestedAvailable )
            {
                m_flatRequestedName.clear();
                m_flatSelectionPending = false;
                flatStopSelectionTimer = true;
            }

            flatStopRefreshTimer = true;
        }

        if( flatStopRefreshTimer )
        {
            emit flatRefreshTimerStop();
        }

        if( flatStopSelectionTimer )
        {
            emit flatSelectionTimerStop();
        }
    }
    else if( ipRecv.getName() == "flat_shmim" )
    {
        if( ipRecv.find( "channel" ) )
        {
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_flatShmim = ipRecv["channel"].get<std::string>();
        }
    }
    else if( ipRecv.getName() == "flat_set" )
    {
        if( ipRecv.find( "toggle" ) )
        {
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( ipRecv["toggle"] == pcf::IndiElement::On )
                m_flatSet = true;
            else
                m_flatSet = false;
        }
    }
    else if( ipRecv.getName() == "test" )
    {
        bool testStopSelectionTimer = false;
        bool testRequestedAvailable = false;
        bool testStopRefreshTimer   = false;

        { // mutex scope
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_testPropertyRefreshing = false;
            m_testOptions.clear();
            m_testName = "";
            for( auto it = ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it )
            {
                m_testOptions.push_back( it->first );
                if( it->first == m_testRequestedName )
                {
                    testRequestedAvailable = true;
                }
                if( ipRecv[it->first] == pcf::IndiElement::On )
                    m_testName = it->first;
            }

            if( m_testSelectionPending && m_testName == m_testRequestedName )
            {
                m_testRequestedName.clear();
                m_testSelectionPending = false;
                testStopSelectionTimer = true;
            }
            else if( m_testSelectionPending && !m_testRequestedName.empty() && !testRequestedAvailable )
            {
                m_testRequestedName.clear();
                m_testSelectionPending = false;
                testStopSelectionTimer = true;
            }

            testStopRefreshTimer = true;
        }

        if( testStopRefreshTimer )
        {
            emit testRefreshTimerStop();
        }

        if( testStopSelectionTimer )
        {
            emit testSelectionTimerStop();
        }
    }
    else if( ipRecv.getName() == "test_shmim" )
    {
        if( ipRecv.find( "channel" ) )
        {
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_testShmim = ipRecv["channel"].get<std::string>();
        }
    }
    else if( ipRecv.getName() == "test_set" )
    {
        if( ipRecv.find( "toggle" ) )
        {
            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( ipRecv["toggle"] == pcf::IndiElement::On )
                m_testSet = true;
            else
                m_testSet = false;
        }
    }

    emit doUpdateGUI();
}

void dmCtrl::updateGUI()
{
    if( m_inUpdate || !m_connected )
        return;

    m_inUpdate = true;

    std::string              appState;
    std::string              shmimName;
    std::string              flatShmim;
    std::string              testShmim;
    bool                     flatSet{ false };
    bool                     testSet{ false };
    std::string              flatName;
    std::string              flatRequestedName;
    std::string              testName;
    std::string              testRequestedName;
    std::vector<std::string> flatOptions;
    std::vector<std::string> testOptions;
    bool                     flatSelectionPending{ false };
    bool                     testSelectionPending{ false };
    bool                     flatPropertyRefreshing{ false };
    bool                     testPropertyRefreshing{ false };

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        appState               = m_appState;
        shmimName              = m_shmimName;
        flatShmim              = m_flatShmim;
        testShmim              = m_testShmim;
        flatSet                = m_flatSet;
        testSet                = m_testSet;
        flatName               = m_flatName;
        flatRequestedName      = m_flatRequestedName;
        testName               = m_testName;
        testRequestedName      = m_testRequestedName;
        flatOptions            = m_flatOptions;
        testOptions            = m_testOptions;
        flatSelectionPending   = m_flatSelectionPending;
        testSelectionPending   = m_testSelectionPending;
        flatPropertyRefreshing = m_flatPropertyRefreshing;
        testPropertyRefreshing = m_testPropertyRefreshing;
    }

    bool flatControlsAvailable = ( !flatPropertyRefreshing && !flatOptions.empty() );
    bool testControlsAvailable = ( !testPropertyRefreshing && !testOptions.empty() );

    ui.labelShmimName_value->setText( shmimName.c_str() );
    ui.labelFlatShmim_value->setText( flatShmim.c_str() );
    ui.labelTestShmim_value->setText( testShmim.c_str() );

    { // mutex scope
        QSignalBlocker blockFlat( ui.comboSelectFlat );

        syncComboOptions( ui.comboSelectFlat, flatOptions );

        std::string flatDisplayName = flatName;
        if( flatSelectionPending && !flatRequestedName.empty() )
        {
            flatDisplayName = flatRequestedName;
        }

        if( !flatDisplayName.empty() )
        {
            int flatIndex = ui.comboSelectFlat->findText( flatDisplayName.c_str() );
            if( flatIndex >= 0 )
                ui.comboSelectFlat->setCurrentIndex( flatIndex );
            else
                ui.comboSelectFlat->setCurrentIndex( -1 );
        }
        else
        {
            ui.comboSelectFlat->setCurrentIndex( -1 );
        }
    }

    { // mutex scope
        QSignalBlocker blockTest( ui.comboSelectTest );

        syncComboOptions( ui.comboSelectTest, testOptions );

        std::string testDisplayName = testName;
        if( testSelectionPending && !testRequestedName.empty() )
        {
            testDisplayName = testRequestedName;
        }

        if( !testDisplayName.empty() )
        {
            int testIndex = ui.comboSelectTest->findText( testDisplayName.c_str() );
            if( testIndex >= 0 )
                ui.comboSelectTest->setCurrentIndex( testIndex );
            else
                ui.comboSelectTest->setCurrentIndex( -1 );
        }
        else
        {
            ui.comboSelectTest->setCurrentIndex( -1 );
        }
    }

    if( appState != "NOTHOMED" && appState != "READY" && appState != "OPERATING" )
    {
        // Disable & zero all

        ui.buttonInit->setEnabled( false );
        ui.buttonZero->setEnabled( false );
        ui.buttonRelease->setEnabled( false );
        ui.comboSelectFlat->setEnabled( false );
        ui.comboSelectTest->setEnabled( false );

        ui.buttonSetFlat->setEnabled( false );
        ui.buttonZeroFlat->setEnabled( false );

        ui.buttonSetTest->setEnabled( false );
        ui.buttonZeroTest->setEnabled( false );

        m_inUpdate = false;
        return;
    }

    if( appState == "NOTHOMED" )
    {

        ui.buttonInit->setEnabled( true );
        ui.buttonZero->setEnabled( false );
        ui.buttonRelease->setEnabled( false );
        ui.comboSelectFlat->setEnabled( false );
        ui.comboSelectTest->setEnabled( false );

        ui.buttonSetFlat->setEnabled( false );
        ui.buttonZeroFlat->setEnabled( false );

        ui.buttonSetTest->setEnabled( false );
        ui.buttonZeroTest->setEnabled( false );

        m_inUpdate = false;
        return;
    }

    ui.buttonInit->setEnabled( false );
    ui.buttonZero->setEnabled( true );
    ui.buttonRelease->setEnabled( true );
    ui.comboSelectFlat->setEnabled( flatControlsAvailable );
    ui.comboSelectTest->setEnabled( testControlsAvailable );

    if( flatControlsAvailable && flatSet == false )
    {
        ui.buttonSetFlat->setEnabled( true );
        ui.buttonZeroFlat->setEnabled( false );
    }
    else if( flatControlsAvailable && flatSet == true )
    {
        ui.buttonSetFlat->setEnabled( false );
        ui.buttonZeroFlat->setEnabled( true );
    }
    else
    {
        ui.buttonSetFlat->setEnabled( false );
        ui.buttonZeroFlat->setEnabled( false );
    }

    if( testControlsAvailable && testSet == false )
    {
        ui.buttonSetTest->setEnabled( true );
        ui.buttonZeroTest->setEnabled( false );
    }
    else if( testControlsAvailable && testSet == true )
    {
        ui.buttonSetTest->setEnabled( false );
        ui.buttonZeroTest->setEnabled( true );
    }
    else
    {
        ui.buttonSetTest->setEnabled( false );
        ui.buttonZeroTest->setEnabled( false );
    }

    m_inUpdate = false;

} // updateGUI()

void dmCtrl::syncComboOptions( QComboBox *combo, const std::vector<std::string> &options )
{
    if( combo == nullptr )
    {
        return;
    }

    bool optionsChanged = ( combo->count() != static_cast<int>( options.size() ) );

    if( !optionsChanged )
    {
        for( int n = 0; n < combo->count(); ++n )
        {
            if( combo->itemText( n ).toStdString() != options[static_cast<size_t>( n )] )
            {
                optionsChanged = true;
                break;
            }
        }
    }

    if( !optionsChanged )
    {
        return;
    }

    combo->clear();
    for( const auto &option : options )
    {
        combo->addItem( option.c_str() );
    }

    updateXwComboBoxPopupWidth( combo );
}

void dmCtrl::on_buttonInit_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "initDM" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void dmCtrl::on_buttonZeroAll_pressed()
{

    pcf::IndiProperty ip( pcf::IndiProperty::Switch );

    ip.setDevice( m_dmName );
    ip.setName( "zeroAll" );
    ip.add( pcf::IndiElement( "request" ) );

    ip["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ip );
}

void dmCtrl::on_buttonZero_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "zeroDM" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void dmCtrl::on_buttonRelease_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "releaseDM" );
    ipFreq.add( pcf::IndiElement( "request" ) );
    ipFreq["request"].setSwitchState( pcf::IndiElement::On );

    sendNewProperty( ipFreq );
}

void dmCtrl::on_comboSelectFlat_activated( int index )
{
    if( index < 0 || index >= ui.comboSelectFlat->count() )
        return;

    std::string       choice = ui.comboSelectFlat->itemText( index ).toStdString();
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );
    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "flat" );

    for( int i = 0; i < ui.comboSelectFlat->count(); ++i )
    {
        std::string eln = ui.comboSelectFlat->itemText( i ).toStdString();
        std::cerr << eln << "\n";
        ipFreq.add( pcf::IndiElement( eln ) );
        if( eln == choice )
            ipFreq[eln] = pcf::IndiElement::On;
        else
            ipFreq[eln] = pcf::IndiElement::Off;
    }

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        m_flatRequestedName    = choice;
        m_flatSelectionPending = true;
    }

    if( m_flatSelectionTimer )
    {
        m_flatSelectionTimer->start( 10000 );
    }

    sendNewProperty( ipFreq );
}

void dmCtrl::on_buttonSetFlat_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "flat_set" );
    ipFreq.add( pcf::IndiElement( "toggle" ) );
    ipFreq["toggle"] = pcf::IndiElement::On;

    sendNewProperty( ipFreq );
}

void dmCtrl::on_buttonZeroFlat_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "flat_set" );
    ipFreq.add( pcf::IndiElement( "toggle" ) );
    ipFreq["toggle"] = pcf::IndiElement::Off;

    sendNewProperty( ipFreq );
}

void dmCtrl::on_comboSelectTest_activated( int index )
{
    if( index < 0 || index >= ui.comboSelectTest->count() )
        return;

    std::string       choice = ui.comboSelectTest->itemText( index ).toStdString();
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );
    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "test" );

    for( int i = 0; i < ui.comboSelectTest->count(); ++i )
    {
        std::string eln = ui.comboSelectTest->itemText( i ).toStdString();
        ipFreq.add( pcf::IndiElement( eln ) );
        if( eln == choice )
            ipFreq[eln] = pcf::IndiElement::On;
        else
            ipFreq[eln] = pcf::IndiElement::Off;
    }

    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        m_testRequestedName    = choice;
        m_testSelectionPending = true;
    }

    if( m_testSelectionTimer )
    {
        m_testSelectionTimer->start( 10000 );
    }

    sendNewProperty( ipFreq );
}

void dmCtrl::on_buttonSetTest_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "test_set" );
    ipFreq.add( pcf::IndiElement( "toggle" ) );
    ipFreq["toggle"] = pcf::IndiElement::On;

    sendNewProperty( ipFreq );
}

void dmCtrl::on_buttonZeroTest_pressed()
{
    pcf::IndiProperty ipFreq( pcf::IndiProperty::Switch );

    ipFreq.setDevice( m_dmName );
    ipFreq.setName( "test_set" );
    ipFreq.add( pcf::IndiElement( "toggle" ) );
    ipFreq["toggle"] = pcf::IndiElement::Off;

    sendNewProperty( ipFreq );
}

void dmCtrl::flatSelectionTimerOut()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        m_flatRequestedName.clear();
        m_flatSelectionPending = false;
    }

    emit doUpdateGUI();
}

void dmCtrl::flatRefreshTimerOut()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        if( !m_flatPropertyRefreshing )
        {
            return;
        }

        m_flatName.clear();
        m_flatOptions.clear();
        m_flatRequestedName.clear();
        m_flatSelectionPending   = false;
        m_flatPropertyRefreshing = false;
    }

    if( m_flatSelectionTimer )
    {
        m_flatSelectionTimer->stop();
    }

    emit doUpdateGUI();
}

void dmCtrl::testSelectionTimerOut()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        m_testRequestedName.clear();
        m_testSelectionPending = false;
    }

    emit doUpdateGUI();
}

void dmCtrl::testRefreshTimerOut()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        if( !m_testPropertyRefreshing )
        {
            return;
        }

        m_testName.clear();
        m_testOptions.clear();
        m_testRequestedName.clear();
        m_testSelectionPending   = false;
        m_testPropertyRefreshing = false;
    }

    if( m_testSelectionTimer )
    {
        m_testSelectionTimer->stop();
    }

    emit doUpdateGUI();
}

} // namespace xqt

#include "moc_dmCtrl.cpp"

#endif
