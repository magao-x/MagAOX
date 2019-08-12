#ifndef xqt_pwrDevice_hpp
#define xqt_pwrDevice_hpp

#include <QWidget>
#include <qwt_text_label.h>

#include <mx/ioutils/stringUtils.hpp>

#include "../../lib/multiIndi.hpp"

#include "pwrChannel.hpp"

inline
double tsDiff( const timespec & ts2,
               const timespec & ts1 
             )
{
   double tsd1 = ((double)ts1.tv_nsec)/1e9;
   double tsd2 = ((double) (ts2.tv_sec - ts1.tv_sec)) + ((double)ts2.tv_nsec)/1e9;
   
   return tsd2-tsd1;
}

template<typename _T>
class circularTimeSeries
{
public:
   typedef _T T;
   
protected:
   
   std::vector<T> m_data; ///< Holds the time series data
   std::vector<timespec> m_timeStamps; ///< Holds the timer series timestamps.
   
   size_t m_currSize {0}; ///< This is the current size of the time series, always <= m_data.size().
   size_t m_currPos {0};  ///< Current position in the circular buffer.
   
public:
   
   circularTimeSeries()
   {
   }
   
   explicit circularTimeSeries(size_t size)
   {
      resize(size);
   }
   
   void resize( size_t size)
   {
      m_data.resize(size, T(0));
      m_timeStamps.resize(size, {0,0});
      
      m_currSize = 0;
      m_currPos = 0;
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
   
   void add( const T & val, const timespec & ts )
   {
      if(m_data.size() == 0)
      {
         resize(1);
      }
      
      
      m_data[m_currPos] = val;
      m_timeStamps[m_currPos] = ts;
      
      ++m_currPos;
      
      //Increase m_currSize up until we reach the full size
      if(m_currSize < m_data.size()) ++m_currSize;
      
      //Wrap 
      if(m_currPos >= m_data.size()) m_currPos = 0;
   }
   
   /// Get the n-th value in the time series
   /** value(0) will return the earliest point currently in the time series.
     * value(currSize()-1) will return the most recently added point.
     */ 
   T value( size_t n )
   {
      n += m_currPos;
      
      if(n >= m_currSize) n = 0;
      
      return m_data[n];
   }
   
   /// Get the n-th timestamp in the time series
   /** timeStamp(0) will return the earliest point currently in the time series.
     * timeStamp(currSize()-1) will return the most recently added point.
     */
   timespec timeStamp( size_t n)
   {
      n += m_currPos;
      
      if(n >= m_data.size()) n = 0;
      
      return m_timeStamps[n];
   }
   
   /// Return the value of the most recent entry in the time series.
   T lastVal()
   {
      size_t n;
      //handle unsigned-ness
      if( m_currSize == 0) return 0;
      n = m_currSize - 1;
      
      return value(n);
   }
   
   /// Return the timestamp of the most recent entry in the time series.
   T lastTimeStamp()
   {
      size_t n;
      //handle unsigned ness
      if( m_currPos == 0) n = m_currSize - 1;
      else n = m_currPos - 1;
      return timeStamp(n);
   }
   
   T averageLast( double avgTime )
   {
      size_t i = m_currSize-1;
      
      double avg = value(i);
      timespec ts0 = timeStamp(i);
      size_t n = 1;
      
      if(i == 0)
      {
         return avg;
      }
      
      --i;
      double dt = 0;
      while( dt <= avgTime )
      {
         dt = tsDiff( ts0, timeStamp(i));
         if(dt < 0) break;

         avg += value(i);
         ++n;
         
         if(i== 0) break;
         --i;
         
      }
      
      return avg/n;
   }
   
};
   
namespace xqt 
{

struct pwrDevice : public QWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_deviceName;

   QwtTextLabel* m_deviceNameLabel {nullptr};
   
   size_t m_numChannels {0};

   pwrChannel ** m_channels {nullptr};
   
   circularTimeSeries<double> m_current;
   circularTimeSeries<double> m_voltage;
   circularTimeSeries<double> m_frequency;
   
public:

   pwrDevice( QWidget * parent = nullptr, 
               Qt::WindowFlags flags = 0
             );

   virtual ~pwrDevice();
   
   std::string deviceName();
   
   void deviceName( const std::string & dname);
   
   void setChannels( const std::vector<std::string> & channelNames);
   
   size_t numChannels();
   
   pwrChannel * channel(size_t channelNo);
   
   QwtTextLabel * deviceNameLabel();
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv );
   
   double current();
   
   double voltage();
   
   double frequency();
   
public slots:

   void switchOn( const std::string & channelName);
   void switchOff( const std::string & channelName);

signals:
   void chChange( pcf::IndiProperty & ip );
   void loadChanged();
   
};

} //namespace xqt 

#endif //xqt_pwrDevice_hpp
