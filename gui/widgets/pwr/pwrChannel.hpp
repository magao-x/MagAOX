/** \file pwrChannel.hpp
 * \brief Power-channel slider state, availability, and command timeout handling.
 */
#ifndef xqt_pwrChannel_hpp
#define xqt_pwrChannel_hpp

#include <string>

#include <QWidget>
#include <QSlider>
#include <QTimer>

#include <qwt_text_label.h>

namespace xqt
{

/// Reported channel states; Unk and unrecognized values disable user control.
enum class pwrChState
{
    Unk,
    Off,
    Int,
    On
};

/// A single power channel control widget
/** Contains the text label and the slider bar control for a single power channel.
 * These widgets are themselves intended to be added to a grid layout -- the
 * pwrChannel widget does not actually manage them.
 */
class pwrChannel : public QWidget
{
    Q_OBJECT

  protected:
    std::string m_channelName; ///< The name of this channel

    QwtTextLabel *m_channelNameLabel{ nullptr }; ///< The widget to display the channel name

    QSlider *m_channelSwitch{ nullptr }; ///< The widget providing user control

    pwrChState m_swTarget{ pwrChState::Unk }; ///< Requested target used to complete a pending command.

    pwrChState m_setSwitchState{ pwrChState::Unk }; ///< Last displayed observation, or Unk while unavailable.

    bool m_changing{ false }; ///< Whether a local command is waiting for its target or timeout.

    std::vector<int> m_outlets; ///< The outlets controlled by this channel.

    double m_onDelay{ 1000 }; ///< The total turn-on delay for this channel (between outlets)

    double m_onTimeout{ 6000 }; ///< Milliseconds to wait for the turn-on target before restoring the observed state.

    double m_offDelay{ 1000 }; ///< The total turn-off delay for this channel (between outlets).

    double m_offTimeout{ 6000 }; ///< Milliseconds to wait for the turn-off target before restoring the observed state.

    bool m_isToggle{ false }; ///< Whether this is a toggle switch (true) or a text switch (false).

    QTimer *m_timer{ nullptr }; ///< Timer for tracking timeouts on channel state changes

  public:
    /// Construct a channel whose slider is disabled until a recognized state arrives.
    /** Creates the label, slider, and timeout timer as children and connects their signals.
     */
    pwrChannel( QWidget        *parent = nullptr, /**< [in] Parent owning this channel widget. */
                Qt::WindowFlags flags  = Qt::WindowFlags() /**< [in] Window flags passed to QWidget. */ );

    /// Destructor
    virtual ~pwrChannel();

    /// Get the channel name
    /**
     * \returns the current value of m_channelName
     */
    std::string channelName();

    /// Set the channel name
    /** Sets m_channelName.
     */
    void channelName( const std::string &nname /**< [in] the new channel name*/ );

    /// Get the slider position as Off (0) or On (2), using the existing threshold.
    int switchState();

    /// Store the target used to complete a pending local command.
    void switchTarget( pwrChState swstate /**< [in] Received target state. */ );

    /// Display a recognized observation or disable the slider for an unknown state.
    /** Unknown observations cancel command waits without moving the slider or confirming the target.
     * Recognized observations restore control subject to the existing pending-command wait.
     */
    void switchState( pwrChState swstate /**< [in] Received observed state. */ );

    /// Check whether a local command is waiting for its target or timeout.
    bool changing();

    /// Get the child label placed in the containing power widget's layout.
    QwtTextLabel *channelNameLabel();

    /// Get the child slider placed in the containing power widget's layout.
    QSlider *channelSwitch();

    /// Store the channel's outlet list and recalculate both command timeouts.
    void outlets( const std::vector<int> &outs /**< [in] Controlled outlet numbers. */ );

    /// Store the total turn-on delay and recalculate its timeout.
    void onDelay( double onD /**< [in] Total delay between outlet turn-on commands in milliseconds. */ );

    /// Store the total turn-off delay and recalculate its timeout.
    void offDelay( double offD /**< [in] Total delay between outlet turn-off commands in milliseconds. */ );

    /// Calculate the turn-on timeout from outlet count and total delay.
    void calcOnTimeout();

    /// Calculate the turn-off timeout from outlet count and total delay.
    void calcOffTimeout();

    /// Select the outgoing command protocol for this channel.
    void isToggle( bool it /**< [in] True for a Switch toggle property, false for Text targets. */ );

    /// Check whether the channel uses a Switch toggle property.
    bool isToggle();

    /// Clear connection metadata and disable the slider until a recognized observation arrives.
    void onDisconnect();

  public slots:

    /// Dispatch a user-requested state change only while the slider is enabled.
    void sliderReleased();

    /// Stop the command timeout and clear the local waiting flag.
    void noTimeOut();

    /// End a command wait and restore the last displayed observation, keeping unknown states disabled.
    void timeOut();

  signals:

    /// Request that the containing device turn this channel on.
    void switchOn( const std::string &channelName /**< [in] Name of the channel to command. */ );

    /// Request that the containing device turn this channel off.
    void switchOff( const std::string &channelName /**< [in] Name of the channel to command. */ );

    /// Notify that a recognized On/Off observation ends the local command wait.
    void switchTargetReached();
};

pwrChannel::pwrChannel( QWidget *parent, Qt::WindowFlags flags ) : QWidget( parent, flags )
{
    m_channelNameLabel = new QwtTextLabel( this );
    m_channelNameLabel->setStyleSheet( "*{color: white;}" );

    m_channelSwitch = new QSlider( this );
    m_channelSwitch->setOrientation( Qt::Horizontal );
    m_channelSwitch->setMinimum( 0 );
    m_channelSwitch->setMaximum( 10 );
    m_channelSwitch->setSingleStep( 1 );
    m_channelSwitch->setPageStep( 1 );
    m_channelSwitch->setEnabled( false );

    QPalette p = m_channelSwitch->palette();
    p.setColor( QPalette::Active, QPalette::Highlight, QColor( 22, 111, 117, 255 ) );   // Scale text and line
    p.setColor( QPalette::Inactive, QPalette::Highlight, QColor( 22, 111, 117, 255 ) ); // Scale text and line
    m_channelSwitch->setPalette( p );

    QObject::connect( m_channelSwitch, SIGNAL( sliderReleased() ), this, SLOT( sliderReleased() ) );

    m_timer = new QTimer( this );
    connect( m_timer, SIGNAL( timeout() ), this, SLOT( timeOut() ) );
    connect( this, SIGNAL( switchTargetReached() ), this, SLOT( noTimeOut() ) );
}

pwrChannel::~pwrChannel()
{
}

std::string pwrChannel::channelName()
{
    return m_channelName;
}

void pwrChannel::channelName( const std::string &nname )
{
    m_channelName = nname;
    m_channelNameLabel->setText( nname.c_str() );
}

int pwrChannel::switchState()
{
    if( m_channelSwitch->sliderPosition() > 0.8 * ( m_channelSwitch->maximum() - m_channelSwitch->minimum() ) )
    {
        return 2;
    }

    return 0;
}

void pwrChannel::switchTarget( pwrChState swstate )
{
    m_swTarget = swstate;
}

void pwrChannel::switchState( pwrChState swstate )
{
    if( swstate != pwrChState::On && swstate != pwrChState::Off && swstate != pwrChState::Int )
    {
        noTimeOut();
        m_setSwitchState = pwrChState::Unk;
        m_channelSwitch->setEnabled( false );
        return;
    }

    if( m_swTarget == pwrChState::Unk )
    {
        m_swTarget = swstate;
    }

    if( swstate != m_swTarget && m_changing )
    {
        m_channelSwitch->setEnabled( false );
        if( swstate == pwrChState::Int )
        {
            m_channelSwitch->setSliderPosition( m_channelSwitch->minimum() +
                                                0.5 * ( m_channelSwitch->maximum() - m_channelSwitch->minimum() ) );
            m_setSwitchState = pwrChState::Int;
        }

        return;
    }

    if( swstate == pwrChState::On )
    {
        m_channelSwitch->setSliderPosition( m_channelSwitch->maximum() );
        m_setSwitchState = pwrChState::On;
        m_channelSwitch->setEnabled( true );
        emit switchTargetReached();
    }
    else if( swstate == pwrChState::Int )
    {
        m_channelSwitch->setSliderPosition( m_channelSwitch->minimum() +
                                            0.5 * ( m_channelSwitch->maximum() - m_channelSwitch->minimum() ) );
        m_setSwitchState = pwrChState::Int;
        m_channelSwitch->setEnabled( true );
    }
    else if( swstate == pwrChState::Off )
    {
        m_channelSwitch->setSliderPosition( m_channelSwitch->minimum() );
        m_setSwitchState = pwrChState::Off;
        m_channelSwitch->setEnabled( true );
        emit switchTargetReached();
    }
}

inline bool pwrChannel::changing()
{
    return m_changing;
}

QwtTextLabel *pwrChannel::channelNameLabel()
{
    return m_channelNameLabel;
}

QSlider *pwrChannel::channelSwitch()
{
    return m_channelSwitch;
}

void pwrChannel::outlets( const std::vector<int> &outs )
{
    m_outlets = outs;

    calcOnTimeout();
    calcOffTimeout();
}

void pwrChannel::onDelay( double onD )
{
    m_onDelay = onD;
    calcOnTimeout();
}

void pwrChannel::offDelay( double offD )
{
    m_offDelay = offD;
    calcOffTimeout();
}

void pwrChannel::calcOnTimeout()
{
    if( m_outlets.size() > 1 )
    {
        m_onTimeout = m_outlets.size() * 5000 + m_onDelay;
    }
    else
    {
        m_onTimeout = 5000 + m_onDelay;
    }
}

void pwrChannel::calcOffTimeout()
{
    if( m_outlets.size() > 1 )
    {
        m_offTimeout = m_outlets.size() * 5000 + m_offDelay;
    }
    else
    {
        m_offTimeout = 5000 + m_offDelay;
    }
}

inline void pwrChannel::isToggle( bool it )
{
    m_isToggle = it;
}

inline bool pwrChannel::isToggle()
{
    return m_isToggle;
}

void pwrChannel::sliderReleased()
{
    if( !m_channelSwitch->isEnabled() )
    {
        return;
    }

    if( m_setSwitchState != pwrChState::On )
    {
        if( m_channelSwitch->sliderPosition() >
            m_channelSwitch->minimum() + 0.8 * ( m_channelSwitch->maximum() - m_channelSwitch->minimum() ) )
        {
            m_channelSwitch->setEnabled( false );
            m_changing = true;
            m_timer->start( m_onTimeout );
            emit switchOn( m_channelName );
        }
        else
        {
            switchState( pwrChState::Off );
        }
    }
    else
    {
        if( m_channelSwitch->sliderPosition() <
            m_channelSwitch->minimum() + 0.2 * ( m_channelSwitch->maximum() - m_channelSwitch->minimum() ) )
        {
            m_channelSwitch->setEnabled( false );
            m_changing = true;
            m_timer->start( m_offTimeout );
            emit switchOff( m_channelName );
        }
        else
        {
            switchState( pwrChState::On );
        }
    }
}

void pwrChannel::noTimeOut()
{
    m_changing = false;
    m_timer->stop();
}

void pwrChannel::timeOut()
{
    m_changing = false;
    m_swTarget = m_setSwitchState;
    switchState( m_setSwitchState );
}

void pwrChannel::onDisconnect()
{
    m_timer->stop();
    m_changing       = false;
    m_swTarget       = pwrChState::Unk;
    m_setSwitchState = pwrChState::Unk;
    m_isToggle       = false;
    m_outlets.clear();
    m_onDelay    = 1000;
    m_onTimeout  = 6000;
    m_offDelay   = 1000;
    m_offTimeout = 6000;

    m_channelSwitch->setEnabled( false );
    m_channelSwitch->setSliderPosition( m_channelSwitch->minimum() );
}

} // namespace xqt

#include "moc_pwrChannel.cpp"

#endif // xqt_pwrChannel_hpp
