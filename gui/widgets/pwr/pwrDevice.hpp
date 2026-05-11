#ifndef xqt_pwrDevice_hpp
#define xqt_pwrDevice_hpp

#include <QWidget>
#include <qwt_text_label.h>

#include <mx/ioutils/stringUtils.hpp>

#include "../../lib/multiIndiSubscriber.hpp"

#include "pwrChannel.hpp"

inline double tsDiff( const timespec &ts2, const timespec &ts1 )
{
    double tsd1 = ( (double)ts1.tv_nsec ) / 1e9;
    double tsd2 = ( (double)( ts2.tv_sec - ts1.tv_sec ) ) + ( (double)ts2.tv_nsec ) / 1e9;

    return tsd2 - tsd1;
}

template <typename _T>
class circularTimeSeries
{
  public:
    typedef _T T;

  protected:
    std::vector<T>        m_data;       ///< Holds the time series data
    std::vector<timespec> m_timeStamps; ///< Holds the timer series timestamps.

    size_t m_currSize{ 0 }; ///< This is the current size of the time series, always <= m_data.size().
    size_t m_currPos{ 0 };  ///< Current position in the circular buffer.

  public:
    circularTimeSeries()
    {
    }

    explicit circularTimeSeries( size_t size )
    {
        resize( size );
    }

    void resize( size_t size )
    {
        m_data.resize( size, T( 0 ) );
        m_timeStamps.resize( size, { 0, 0 } );

        m_currSize = 0;
        m_currPos  = 0;
    }

    /// Get the current size of the time-series.
    /** This is not necessarily m_data.size(), if the
     * full number of points have not been added yet after
     * the last resize.
     *
     * To check m_data.size() use capacity().
     *
     * \returns the value of m_currSize, the number of points currently stored in the time-series.
     */
    size_t size()
    {
        return m_currSize;
    }

    /// Get the allocated size of the circular buffer.
    /** This is not necessarily the number of points added,
     * for that use size().
     *
     * \returns m_data.size()
     *
     */
    size_t capacity()
    {
        return m_data.size();
    }

    void add( const T &val, const timespec &ts )
    {
        if( m_data.size() == 0 )
        {
            resize( 1 );
        }

        m_data[m_currPos]       = val;
        m_timeStamps[m_currPos] = ts;

        ++m_currPos;

        // Increase m_currSize up until we reach the full size
        if( m_currSize < m_data.size() )
            ++m_currSize;

        // Wrap
        if( m_currPos >= m_data.size() )
            m_currPos = 0;
    }

    /// Get the n-th value in the time series
    /** value(0) will return the earliest point currently in the time series.
     * value(currSize()-1) will return the most recently added point.
     */
    T value( size_t n )
    {
        n += m_currPos;

        if( n >= m_currSize )
            n = 0;

        return m_data[n];
    }

    /// Get the n-th timestamp in the time series
    /** timeStamp(0) will return the earliest point currently in the time series.
     * timeStamp(currSize()-1) will return the most recently added point.
     */
    timespec timeStamp( size_t n )
    {
        n += m_currPos;

        if( n >= m_data.size() )
            n = 0;

        return m_timeStamps[n];
    }

    /// Return the value of the most recent entry in the time series.
    T lastVal()
    {
        size_t n;
        // handle unsigned-ness
        if( m_currSize == 0 )
            return 0;
        n = m_currSize - 1;

        return value( n );
    }

    /// Return the timestamp of the most recent entry in the time series.
    T lastTimeStamp()
    {
        size_t n;
        // handle unsigned ness
        if( m_currPos == 0 )
            n = m_currSize - 1;
        else
            n = m_currPos - 1;
        return timeStamp( n );
    }

    T averageLast( double avgTime )
    {
        size_t i = m_currSize - 1;

        double   avg = value( i );
        timespec ts0 = timeStamp( i );
        size_t   n   = 1;

        if( i == 0 )
        {
            return avg;
        }

        --i;
        double dt = 0;
        while( dt <= avgTime )
        {
            dt = tsDiff( ts0, timeStamp( i ) );
            if( dt < 0 )
                break;

            avg += value( i );
            ++n;

            if( i == 0 )
                break;
            --i;
        }

        return avg / n;
    }
};

namespace xqt
{

struct pwrDevice : public QWidget
{
    Q_OBJECT

  protected:
    std::string m_deviceName;

    QwtTextLabel *m_deviceNameLabel{ nullptr };

    size_t m_numChannels{ 0 };

    pwrChannel **m_channels{ nullptr };

    circularTimeSeries<double> m_current;
    circularTimeSeries<double> m_voltage;
    circularTimeSeries<double> m_frequency;

  public:
    pwrDevice( QWidget *parent = nullptr, Qt::WindowFlags flags = Qt::WindowFlags() );

    virtual ~pwrDevice();

    std::string deviceName() const;

    void deviceName( const std::string &dname );

    void setChannels( const std::vector<std::string> &channelNames );

    size_t numChannels();

    pwrChannel *channel( size_t channelNo );

    QwtTextLabel *deviceNameLabel();

    void onDisconnect();

    void handleDelProperty( const pcf::IndiProperty &ipRecv );

    void handleSetProperty( const pcf::IndiProperty &ipRecv );

    double current();

    double voltage();

    double frequency();

  public slots:

    void switchOn( const std::string &channelName );
    void switchOff( const std::string &channelName );

  signals:
    void chChange( pcf::IndiProperty &ip );
    void loadChanged();
};

inline bool compPwrDevice( const pwrDevice *one, const pwrDevice *two )
{
    return ( one->deviceName() < two->deviceName() );
}

pwrDevice::pwrDevice( QWidget *parent, Qt::WindowFlags flags ) : QWidget( parent, flags )
{
    m_deviceNameLabel = new QwtTextLabel;
    m_deviceNameLabel->setStyleSheet( "*{color: white;}" );

    m_current.resize( 60 );
    m_voltage.resize( 60 );
    m_frequency.resize( 60 );
}

pwrDevice::~pwrDevice()
{

    if( m_numChannels > 0 )
    {
        for( size_t i = 0; i < m_numChannels; ++i )
        {
            m_channels[i]->deleteLater();
        }
    }

    if( m_channels )
    {
        delete[] m_channels;
    }

    // This is taken care of by parent destruct:
    // delete m_deviceNameLabel;
}

std::string pwrDevice::deviceName() const
{
    return m_deviceName;
}

void pwrDevice::deviceName( const std::string &dname )
{
    m_deviceName = dname;

    m_deviceNameLabel->setText( m_deviceName.c_str() );
}

void pwrDevice::setChannels( const std::vector<std::string> &channelNames )
{
    if( m_numChannels > 0 )
    {
        for( size_t i = 0; i < m_numChannels; ++i )
        {
            m_channels[i]->deleteLater();
        }
    }

    if( m_channels )
    {
        delete[] m_channels;
    }

    m_channels = nullptr;

    m_numChannels = channelNames.size();
    if( m_numChannels == 0 )
    {
        return;
    }

    m_channels = new pwrChannel *[m_numChannels];

    for( size_t i = 0; i < m_numChannels; ++i )
    {
        m_channels[i] = new pwrChannel;
        m_channels[i]->channelName( channelNames[i] );
        QObject::connect(
            m_channels[i], SIGNAL( switchOn( const std::string & ) ), this, SLOT( switchOn( const std::string & ) ) );
        QObject::connect(
            m_channels[i], SIGNAL( switchOff( const std::string & ) ), this, SLOT( switchOff( const std::string & ) ) );
    }

    return;
}

size_t pwrDevice::numChannels()
{
    return m_numChannels;
}

pwrChannel *pwrDevice::channel( size_t channelNo )
{
    if( channelNo >= m_numChannels )
        return nullptr;

    return m_channels[channelNo];
}

QwtTextLabel *pwrDevice::deviceNameLabel()
{
    return m_deviceNameLabel;
}

void pwrDevice::onDisconnect()
{
    m_current.resize( 60 );
    m_voltage.resize( 60 );
    m_frequency.resize( 60 );

    for( size_t i = 0; i < m_numChannels; ++i )
    {
        m_channels[i]->onDisconnect();
    }
}

void pwrDevice::handleDelProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != deviceName() )
        return;

    if( ipRecv.getName() == "load" )
    {
        m_current.resize( 60 );
        m_voltage.resize( 60 );
        m_frequency.resize( 60 );
        emit loadChanged();
        return;
    }

    if( ipRecv.getName() == "channelOutlets" || ipRecv.getName() == "channelOnDelays" || ipRecv.getName() == "channelOffDelays" )
    {
        for( size_t i = 0; i < m_numChannels; ++i )
        {
            m_channels[i]->onDisconnect();
        }
        return;
    }

    for( size_t i = 0; i < m_numChannels; ++i )
    {
        if( ipRecv.getName() == m_channels[i]->channelName() )
        {
            m_channels[i]->onDisconnect();
            return;
        }
    }
}

void pwrDevice::handleSetProperty( const pcf::IndiProperty &ipRecv )
{
    if( ipRecv.getDevice() != deviceName() )
        return;

    if( ipRecv.getName() == "channelOutlets" )
    {
        for( size_t n = 0; n < m_numChannels; ++n )
        {
            if( ipRecv.find( m_channels[n]->channelName() ) )
            {

                std::string outletStr = ipRecv[m_channels[n]->channelName()].get();

                std::vector<int> outlets;
                mx::ioutils::parseStringVector( outlets, outletStr );

                m_channels[n]->outlets( outlets );
            }
        }

        return;
    }

    if( ipRecv.getName() == "channelOnDelays" )
    {
        for( size_t n = 0; n < m_numChannels; ++n )
        {
            if( ipRecv.find( m_channels[n]->channelName() ) )
            {
                double onDelay = ipRecv[m_channels[n]->channelName()].get<double>();
                m_channels[n]->onDelay( onDelay );
            }
        }

        return;
    }

    if( ipRecv.getName() == "channelOffDelays" )
    {
        for( size_t n = 0; n < m_numChannels; ++n )
        {
            if( ipRecv.find( m_channels[n]->channelName() ) )
            {
                double offDelay = ipRecv[m_channels[n]->channelName()].get<double>();
                m_channels[n]->offDelay( offDelay );
            }
        }

        return;
    }

    // Check for state
    for( size_t i = 0; i < m_numChannels; ++i )
    {
        if( ipRecv.getName() == m_channels[i]->channelName() )
        {
            if( ipRecv.getType() == pcf::IndiProperty::Switch )
            {
                if( ipRecv.find( "toggle" ) )
                {
                    m_channels[i]->isToggle( true );
                    if( ipRecv.getState() == pcf::IndiProperty::Busy )
                    {
                        m_channels[i]->switchState( pwrChState::Int );
                    }
                    else if( ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On )
                    {
                        m_channels[i]->switchState( pwrChState::On );
                    }
                    else
                    {
                        m_channels[i]->switchState( pwrChState::Off );
                    }
                }
            }
            else
            {
                if( ipRecv.find( "target" ))
                {
                    std::string target = ipRecv["target"].get();

                    if( target == "Int" )
                    {
                        m_channels[i]->switchTarget( pwrChState::Int );
                    }
                    else if( target == "On" )
                    {
                        m_channels[i]->switchTarget( pwrChState::On );
                    }
                    else if( target == "Off" )
                    {
                        m_channels[i]->switchTarget( pwrChState::Off );
                    }
                }

                if( ipRecv.find( "state" ))
                {
                    std::string state = ipRecv["state"].get();

                    if( state == "On" )
                    {
                        m_channels[i]->switchState( pwrChState::On );
                    }
                    else if( state == "Int" )
                    {
                        m_channels[i]->switchState( pwrChState::Int );
                    }
                    else if( state == "Off" )
                    {
                        m_channels[i]->switchState( pwrChState::Off );
                    }
                }
            }
        }
    }

    if( ipRecv.getName() == "load" )
    {
        timespec ts;
        clock_gettime( CLOCK_REALTIME, &ts );

        if( ipRecv.find( "current" ) )
        {
            m_current.add( ipRecv["current"].get<double>(), ts );
        }

        if( ipRecv.find( "voltage" ) )
        {
            m_voltage.add( ipRecv["voltage"].get<double>(), ts );
        }

        if( ipRecv.find( "frequency" ) )
        {
            m_frequency.add( ipRecv["frequency"].get<double>(), ts );
        }

        emit loadChanged();
    }
}

double pwrDevice::current()
{
    if( m_current.size() == 0 )
        return -1;
    return m_current.lastVal();
}

double pwrDevice::voltage()
{
    if( m_voltage.size() == 0 )
        return -1;
    return m_voltage.averageLast( 10 );
}

double pwrDevice::frequency()
{
    if( m_frequency.size() == 0 )
        return -1;
    return m_frequency.averageLast( 10 );
}

void pwrDevice::switchOn( const std::string &channelName )
{
    bool toggle = false;
    for( size_t n = 0; n < m_numChannels; ++n )
    {
        if( m_channels[n]->channelName() == channelName )
        {
            toggle = m_channels[n]->isToggle();
            break;
        }
    }

    if( toggle )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );

        ip.setDevice( m_deviceName );
        ip.setName( channelName );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"] = pcf::IndiElement::On;

        emit chChange( ip );
    }
    else
    {

        pcf::IndiProperty ip( pcf::IndiProperty::Text );

        ip.setDevice( m_deviceName );
        ip.setName( channelName );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"] = "On";

        emit chChange( ip );
    }
}

void pwrDevice::switchOff( const std::string &channelName )
{
    bool toggle = false;
    for( size_t n = 0; n < m_numChannels; ++n )
    {
        if( m_channels[n]->channelName() == channelName )
        {
            toggle = m_channels[n]->isToggle();
            break;
        }
    }

    if( toggle )
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Switch );

        ip.setDevice( m_deviceName );
        ip.setName( channelName );
        ip.add( pcf::IndiElement( "toggle" ) );
        ip["toggle"] = pcf::IndiElement::Off;

        emit chChange( ip );
    }
    else
    {
        pcf::IndiProperty ip( pcf::IndiProperty::Text );

        ip.setDevice( m_deviceName );
        ip.setName( channelName );
        ip.add( pcf::IndiElement( "target" ) );
        ip["target"] = "Off";

        emit chChange( ip );
    }
}

} // namespace xqt

#include "moc_pwrDevice.cpp"

#endif // xqt_pwrDevice_hpp
