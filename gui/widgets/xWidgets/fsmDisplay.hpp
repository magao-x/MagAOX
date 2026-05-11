/** \file fsmDisplay.hpp
 * \brief Widget that displays and highlights FSM state from an INDI property.
 */
#ifndef fsmDisplay_hpp
#define fsmDisplay_hpp

#include <mutex>

#include "xWidget.hpp"
#include "statusLabel.hpp"

#include "ui_fsmDisplay.h"

namespace xqt
{

class fsmDisplay : public xWidget
{
    Q_OBJECT

  protected:
    std::string m_device;             ///< INDI device name this display subscribes to.
    std::string m_property{ "fsm" };  ///< INDI property name used for FSM state.
    std::string m_element{ "state" }; ///< INDI element name containing FSM state text.

    bool m_highlightChanges{ true }; ///< Enables changed-state text highlighting behavior.

    bool m_valChanged{ false }; ///< True when cached value differs from last displayed value.

    std::string m_value;      ///< Cached display value mirrored from INDI updates.
    std::mutex  m_stateMutex; ///< Guards value cache shared between callback and GUI thread.

    std::string m_NOTHOMED{ "NOTHOMED" };   ///< Display label mapping for NOTHOMED state.
    std::string m_HOMING{ "HOMING" };       ///< Display label mapping for HOMING state.
    std::string m_READY{ "READY" };         ///< Display label mapping for READY state.
    std::string m_OPERATING{ "OPERATING" }; ///< Display label mapping for OPERATING state.

  public:
    /// Constructs the FSM display widget.
    fsmDisplay( QWidget        *Parent = 0,                /**< [in] Optional parent widget. */
                Qt::WindowFlags f      = Qt::WindowFlags() /**< [in] Qt window flags. */
    );

    /// Constructs the FSM display widget with initial target device.
    explicit fsmDisplay( const std::string &device,                    /**< [in] INDI device name. */
                         QWidget           *Parent = 0,                /**< [in] Optional parent widget. */
                         Qt::WindowFlags    f      = Qt::WindowFlags() /**< [in] Qt window flags. */
    );

    /// Destroys the FSM display widget.
    ~fsmDisplay();

    /// Sets the target INDI device for subscription.
    void device( const std::string &dev /**< [in] INDI device name. */ );

    /// Subscribes to configured FSM property updates.
    virtual void subscribe();

    /// Handles parent connection notifications.
    virtual void onConnect();

    /// Handles parent disconnection notifications.
    virtual void onDisconnect();

    /// Clears keyboard focus on this widget.
    virtual void clearFocus();

    /// Handles defProperty notifications.
    virtual void handleDefProperty( const pcf::IndiProperty &ipRecv /**< [in] Property which has changed. */ );

    /// Handles delProperty notifications.
    virtual void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Property which has been deleted. */ );

    /// Handles setProperty notifications.
    virtual void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Property which has changed. */ );

    /// Overrides text shown for NOTHOMED state.
    void NOTHOMED( const std::string &s /**< [in] Replacement label text. */ );

    /// Overrides text shown for HOMING state.
    void HOMING( const std::string &s /**< [in] Replacement label text. */ );

    /// Overrides text shown for READY state.
    void READY( const std::string &s /**< [in] Replacement label text. */ );

    /// Overrides text shown for OPERATING state.
    void OPERATING( const std::string &s /**< [in] Replacement label text. */ );

  public slots:
    /// Applies connected-state GUI behavior on widget thread.
    void onConnectGUI();

    /// Applies disconnected-state GUI behavior on widget thread.
    void onDisconnectGUI();

    /// Refreshes display from cached state.
    void updateGUI();

  signals:
    /// Queues connected-state GUI update onto widget thread.
    void doOnConnect();

    /// Queues disconnected-state GUI update onto widget thread.
    void doOnDisconnect();

    /// Queues display refresh onto widget thread.
    void doUpdateGUI();

  private:
    Ui::fsmDisplay ui; ///< Generated Qt UI object.
};

fsmDisplay::fsmDisplay( QWidget *Parent, Qt::WindowFlags f ) : xWidget( Parent, f )
{
    ui.setupUi( this );

    connect( this, SIGNAL( doOnConnect() ), this, SLOT( onConnectGUI() ), Qt::QueuedConnection );
    connect( this, SIGNAL( doOnDisconnect() ), this, SLOT( onDisconnectGUI() ), Qt::QueuedConnection );
    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ), Qt::QueuedConnection );

    onDisconnectGUI();
}

fsmDisplay::fsmDisplay( const std::string &device, QWidget *Parent, Qt::WindowFlags f )
    : xWidget( Parent, f ), m_device{ device }
{
    ui.setupUi( this );
    connect( this, SIGNAL( doOnConnect() ), this, SLOT( onConnectGUI() ), Qt::QueuedConnection );
    connect( this, SIGNAL( doOnDisconnect() ), this, SLOT( onDisconnectGUI() ), Qt::QueuedConnection );
    connect( this, SIGNAL( doUpdateGUI() ), this, SLOT( updateGUI() ), Qt::QueuedConnection );
    onDisconnectGUI();
}

void fsmDisplay::device( const std::string &dev )
{
    m_device = dev;
}

fsmDisplay::~fsmDisplay()
{
}

void fsmDisplay::subscribe()
{
    if( !m_parent )
        return;

    if( m_property != "" )
        m_parent->addSubscriberProperty( this, m_device, m_property );

    return;
}

void fsmDisplay::onConnect()
{
    emit doOnConnect();
}

void fsmDisplay::onConnectGUI()
{
    std::lock_guard<std::mutex> lock( m_stateMutex );
    m_valChanged = true;
}

void fsmDisplay::onDisconnect()
{
    emit doOnDisconnect();
}

void fsmDisplay::onDisconnectGUI()
{
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        m_value      = "---";
        m_valChanged = false;
    }
    ui.fsm->setText( "---" );
}

void fsmDisplay::handleDefProperty( const pcf::IndiProperty &ipRecv )
{
    return handleSetProperty( ipRecv );
}

void fsmDisplay::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    if( ipRecv.getName() == m_property )
    {
        onDisconnect();
    }
}

void fsmDisplay::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != m_device )
        return;

    if( ipRecv.getName() == m_property )
    {
        if( ipRecv.find( m_element ) )
        {
            std::string value = ipRecv[m_element].get();

            if( value == "NOTHOMED" )
            {
                value = m_NOTHOMED;
            }
            else if( value == "HOMING" )
            {
                value = m_HOMING;
            }
            else if( value == "READY" )
            {
                value = m_READY;
            }
            else if( value == "OPERATING" )
            {
                value = m_OPERATING;
            }

            std::lock_guard<std::mutex> lock( m_stateMutex );
            if( value != m_value )
                m_valChanged = true;
            m_value = value;
        }
    }

    emit doUpdateGUI();
}

void fsmDisplay::NOTHOMED( const std::string &s )
{
    m_NOTHOMED = s;
}

void fsmDisplay::HOMING( const std::string &s )
{
    m_HOMING = s;
}

void fsmDisplay::READY( const std::string &s )
{
    m_READY = s;
}

void fsmDisplay::OPERATING( const std::string &s )
{
    m_OPERATING = s;
}

void fsmDisplay::updateGUI()
{
    bool        valChanged{ false };
    std::string valueStr;
    { // mutex scope
        std::lock_guard<std::mutex> lock( m_stateMutex );
        valChanged = m_valChanged;
        valueStr   = m_value;
    }

    if( isEnabled() )
    {
        if( valChanged )
        {
            QString value( valueStr.c_str() ); // in future provide translatiosn for "RIP" "MODULATING", etc.
            ui.fsm->setTextChanged( value );
            std::lock_guard<std::mutex> lock( m_stateMutex );
            m_valChanged = false;
        }
    }
    else
    {
        QString value( valueStr.c_str() ); // in future provide translatiosn for "RIP" "MODULATING", etc.
        ui.fsm->setText( value );
    }

} // updateGUI()

void fsmDisplay::clearFocus()
{
}

} // namespace xqt

#include "moc_fsmDisplay.cpp"

#endif
