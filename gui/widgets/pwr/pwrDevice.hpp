/** \file pwrDevice.hpp
 * \brief INDI power-device channel controls and electrical sample history.
 */
#ifndef xqt_pwrDevice_hpp
#define xqt_pwrDevice_hpp

#include <QWidget>
#include <qwt_text_label.h>

#include <mx/ioutils/stringUtils.hpp>

#include "../../lib/multiIndiSubscriber.hpp"

#include "pwrChannel.hpp"

/// Return the elapsed seconds between two timestamps.
inline double tsDiff( const timespec &ts2, /**< [in] Later timestamp. */
                      const timespec &ts1 /**< [in] Earlier timestamp. */ )
{
    double tsd1 = ( (double)ts1.tv_nsec ) / 1e9;
    double tsd2 = ( (double)( ts2.tv_sec - ts1.tv_sec ) ) + ( (double)ts2.tv_nsec ) / 1e9;

    return tsd2 - tsd1;
}

/// Circular sample storage used by the power GUI's electrical gauges.
template <typename _T>
class circularTimeSeries
{
  public:
    /// Type of each stored sample.
    typedef _T T;

  protected:
    std::vector<T>        m_data;       ///< Holds the time series data
    std::vector<timespec> m_timeStamps; ///< Timestamps corresponding to the stored samples.

    size_t m_currSize{ 0 }; ///< This is the current size of the time series, always <= m_data.size().
    size_t m_currPos{ 0 };  ///< Position where the next sample will be written.

  public:
    /// Construct an empty sample buffer.
    circularTimeSeries();

    /// Construct a buffer with the requested capacity.
    explicit circularTimeSeries( size_t size /**< [in] Number of sample slots to allocate. */ );

    /// Resize sample storage and reset the sample count and write position.
    void resize( size_t size /**< [in] Number of sample slots to allocate. */ );

    /// Get the current size of the time-series.
    /** This is not necessarily m_data.size(), if the
     * full number of points have not been added yet after
     * the last resize.
     *
     * To check m_data.size() use capacity().
     *
     * \returns the value of m_currSize, the number of points currently stored in the time-series.
     */
    size_t size();

    /// Get the allocated size of the circular buffer.
    /** This is not necessarily the number of points added,
     * for that use size().
     *
     * \returns m_data.size()
     *
     */
    size_t capacity();

    /// Append a sample and its timestamp, replacing the oldest slot when full.
    void add( const T        &val, /**< [in] Value to store. */
              const timespec &ts /**< [in] Time at which the value was sampled. */ );

    /// Read a stored value using the existing circular-buffer indexing.
    T value( size_t n /**< [in] Position relative to the circular-buffer cursor. */ );

    /// Read a stored timestamp using the existing circular-buffer indexing.
    timespec timeStamp( size_t n /**< [in] Position relative to the circular-buffer cursor. */ );

    /// Return the value of the most recent entry in the time series.
    T lastVal();

    /// Return the timestamp of the most recent entry in the time series.
    T lastTimeStamp();

    /// Average the recent sample window using the existing buffer indexing.
    /** Requires at least one stored sample.
     */
    T averageLast( double avgTime /**< [in] Width of the averaging window in seconds. */ );
};

template <typename _T>
inline circularTimeSeries<_T>::circularTimeSeries()
{
}

template <typename _T>
inline circularTimeSeries<_T>::circularTimeSeries( size_t size )
{
    resize( size );
}

template <typename _T>
inline void circularTimeSeries<_T>::resize( size_t size )
{
    m_data.resize( size, T( 0 ) );
    m_timeStamps.resize( size, { 0, 0 } );

    m_currSize = 0;
    m_currPos  = 0;
}

template <typename _T>
inline size_t circularTimeSeries<_T>::size()
{
    return m_currSize;
}

template <typename _T>
inline size_t circularTimeSeries<_T>::capacity()
{
    return m_data.size();
}

template <typename _T>
inline void circularTimeSeries<_T>::add( const T &val, const timespec &ts )
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

template <typename _T>
inline _T circularTimeSeries<_T>::value( size_t n )
{
    n += m_currPos;

    if( n >= m_currSize )
        n = 0;

    return m_data[n];
}

template <typename _T>
inline timespec circularTimeSeries<_T>::timeStamp( size_t n )
{
    n += m_currPos;

    if( n >= m_data.size() )
        n = 0;

    return m_timeStamps[n];
}

template <typename _T>
inline _T circularTimeSeries<_T>::lastVal()
{
    size_t n;
    // handle unsigned-ness
    if( m_currSize == 0 )
        return 0;
    n = m_currSize - 1;

    return value( n );
}

template <typename _T>
inline _T circularTimeSeries<_T>::lastTimeStamp()
{
    size_t n;
    // handle unsigned ness
    if( m_currPos == 0 )
        n = m_currSize - 1;
    else
        n = m_currPos - 1;
    return timeStamp( n );
}

template <typename _T>
inline _T circularTimeSeries<_T>::averageLast( double avgTime )
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

namespace xqt
{

/// Power device whose channels and electrical measurements are updated from INDI properties.
struct pwrDevice : public QWidget
{
    Q_OBJECT

  protected:
    std::string m_deviceName; ///< INDI device whose properties update this control group.

    QwtTextLabel *m_deviceNameLabel{ nullptr }; ///< Label placed in the containing power widget's layout.

    size_t m_numChannels{ 0 }; ///< Number of configured channel widgets.

    pwrChannel **m_channels{
        nullptr }; ///< Owned pointer array; channel widgets are deleted when replaced or destroyed.

    circularTimeSeries<double> m_current; ///< Current samples for the device load gauge.

    circularTimeSeries<double> m_voltage; ///< Voltage samples averaged for the device load gauge.

    circularTimeSeries<double> m_frequency; ///< Frequency samples averaged for the device load gauge.

  public:
    /// Construct a device label and empty electrical sample histories.
    pwrDevice( QWidget        *parent = nullptr, /**< [in] Parent owning this device widget. */
               Qt::WindowFlags flags  = Qt::WindowFlags() /**< [in] Window flags passed to QWidget. */ );

    /// Release channel storage and schedule its widgets for deletion.
    virtual ~pwrDevice();

    /// Get the subscribed INDI device name.
    std::string deviceName() const;

    /// Set the subscribed INDI device name and its displayed label.
    void deviceName( const std::string &dname /**< [in] INDI device name. */ );

    /// Replace the channel widgets and connect their command signals.
    void setChannels( const std::vector<std::string> &channelNames /**< [in] Channel property names to display. */ );

    /// Get the number of configured channels.
    size_t numChannels();

    /// Get a channel widget, or nullptr if the index is outside the configured range.
    pwrChannel *channel( size_t channelNo /**< [in] Zero-based channel index. */ );

    /// Get the device label placed in the containing power widget's layout.
    QwtTextLabel *deviceNameLabel();

    /// Clear electrical histories and disable every channel on disconnection.
    void onDisconnect();

    /// Clear measurements or disable channels whose properties have been deleted.
    void handleDelProperty( const pcf::IndiProperty &ipRecv /**< [in] Deleted INDI property. */ );

    /// Apply channel metadata, observed states, targets, or electrical measurements.
    /** Unk and unrecognized channel state strings disable their slider; target-only updates preserve availability.
     */
    void handleSetProperty( const pcf::IndiProperty &ipRecv /**< [in] Received INDI property update. */ );

    /// Get the current sample, or -1 if no measurement is available.
    double current();

    /// Get the ten-second voltage average, or -1 if no measurement is available.
    double voltage();

    /// Get the ten-second frequency average, or -1 if no measurement is available.
    double frequency();

  public slots:

    /// Emit an On command using the selected channel's Text or Switch protocol.
    void switchOn( const std::string &channelName /**< [in] Channel property to command. */ );

    /// Emit an Off command using the selected channel's Text or Switch protocol.
    void switchOff( const std::string &channelName /**< [in] Channel property to command. */ );

  signals:
    /// Pass a constructed channel command to the containing power widget.
    void chChange( pcf::IndiProperty &ip /**< [in] Outgoing INDI property. */ );

    /// Notify that displayed electrical measurements changed.
    void loadChanged();
};

/// Order power devices by their INDI names.
inline bool compPwrDevice( const pwrDevice *one, /**< [in] First device to compare. */
                           const pwrDevice *two /**< [in] Second device to compare. */ )
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

    if( ipRecv.getName() == "channelOutlets" || ipRecv.getName() == "channelOnDelays" ||
        ipRecv.getName() == "channelOffDelays" )
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
                if( ipRecv.find( "target" ) )
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

                if( ipRecv.find( "state" ) )
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
                    else
                    {
                        m_channels[i]->switchState( pwrChState::Unk );
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
