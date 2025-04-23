
#ifndef xqt_pwrChannel_hpp
#define xqt_pwrChannel_hpp

#include <string>

#include <QWidget>
#include <QSlider>
#include <QTimer>

#include <qwt_text_label.h>

namespace xqt
{

enum class pwrChState{ Unk, Off, Int, On};

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

    pwrChState m_swTarget{ pwrChState::Unk };

    pwrChState m_setSwitchState{ pwrChState::Off }; ///< The last state set by the user.

    bool m_changing {false}; ///< Flag tracking if this channel is changing

    std::vector<int> m_outlets; ///< The outlets controlled by this channel.

    double m_onDelay{ 1000 }; ///< The total turn-on delay for this channel (between outlets)

    double m_onTimeout{ 6000 }; /**< The turn-ontimeout for this channel, the time to wait for device to update the
                                      status before re-enabling the switch.*/

    double m_offDelay{ 1000 }; ///> The turn-off delay for this channel (between outlets)

    double m_offTimeout{ 6000 }; /**< The turn-off timeout for this channel, the time to wait for device to update the
                                      status before re-enabling the switch.*/

    bool m_isToggle {false}; ///< Whether this is a toggle switch (true) or a text switch (false).

    QTimer *m_timer{ nullptr }; ///< Timer for tracking timeouts on channel state changes

  public:
    /// Constructor
    /** Constructs the m_channelNameLabel and m_channelSwitch widgets, sets the palette of m_channelSwitch, an connects
     * the m_channelSwitch sliderReleased signal to the sliderRelased slot.
     */
    pwrChannel( QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags() );

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

    int switchState();

    void switchTarget( pwrChState swstate );

    void switchState( pwrChState swstate );

    bool changing()
    {
        return m_changing;
    }

    QwtTextLabel *channelNameLabel();

    QSlider *channelSwitch();

    void outlets( const std::vector<int> &outs );

    void onDelay( double onD );

    void offDelay( double offD );

    void calcOnTimeout();

    void calcOffTimeout();

    void isToggle(bool it)
    {
        m_isToggle = it;
    }

    bool isToggle()
    {
        return m_isToggle;
    }

  public slots:

    void sliderReleased();

    void noTimeOut();

    void timeOut();

  signals:

    void switchOn( const std::string &channelName );

    void switchOff( const std::string &channelName );

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
    if( m_swTarget == pwrChState::Unk )
    {
        m_swTarget = swstate;
    }

    if( swstate != m_swTarget && m_changing)
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
    }
    else
    {
        m_channelSwitch->setSliderPosition( m_channelSwitch->minimum() );
        m_setSwitchState = pwrChState::Off;
        m_channelSwitch->setEnabled( true );
        emit switchTargetReached();
    }
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
    if(m_outlets.size() > 1)
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
    if(m_outlets.size() > 1)
    {
        m_offTimeout = m_outlets.size() * 5000 + m_offDelay;
    }
    else
    {
        m_offTimeout = 5000 + m_offDelay;
    }
}

void pwrChannel::sliderReleased()
{
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

} // namespace xqt

#include "moc_pwrChannel.cpp"

#endif // xqt_pwrChannel_hpp
