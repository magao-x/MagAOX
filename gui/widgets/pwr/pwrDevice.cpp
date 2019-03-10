
#include "pwrDevice.hpp"

#include <iostream>

namespace xqt
{
   
pwrDevice::pwrDevice( QWidget * parent, 
                        Qt::WindowFlags flags
                      ) : QWidget(parent, flags)
{
   m_deviceNameLabel = new QwtTextLabel;
   
   m_current.resize(60);
   m_voltage.resize(60);
   m_frequency.resize(60);
   
}

pwrDevice::~pwrDevice()
{
   if(m_numChannels > 0)
   {
      for(size_t i=0; i<m_numChannels; ++i)
      {
         delete m_channels[i];
      }
   }
   
   if(m_channels) delete[] m_channels;
   
   delete m_deviceNameLabel;
}
   
std::string pwrDevice::deviceName()
{
   return m_deviceName;
}

void pwrDevice::deviceName( const std::string & dname)
{
   m_deviceName = dname;
   
   m_deviceNameLabel->setText(m_deviceName.c_str());
}

   
void pwrDevice::setChannels( const std::vector<std::string> & channelNames)
{
   if(m_numChannels > 0)
   {
      for(size_t i=0; i<m_numChannels; ++i)
      {
         delete m_channels[i];
      }
   }
   
   if(m_channels) delete[] m_channels;
   
   m_numChannels = channelNames.size();
   if(m_numChannels == 0)
   {
      m_channels = nullptr;
      return;
   }
   
   m_channels = new pwrChannel*[m_numChannels];
   
   for(size_t i=0; i<m_numChannels; ++i)
   {
      m_channels[i] = new pwrChannel;
      m_channels[i]->channelName(channelNames[i]);
      QObject::connect( m_channels[i], SIGNAL(switchOn(const std::string &)), this, SLOT(switchOn(const std::string &)));
      QObject::connect( m_channels[i], SIGNAL(switchOff(const std::string &)), this, SLOT(switchOff(const std::string &)));
   }
   
   return;
}

size_t pwrDevice::numChannels()
{
   return m_numChannels;
}

pwrChannel * pwrDevice::channel(size_t channelNo)
{
   if(channelNo >= m_numChannels) return nullptr;
   
   return m_channels[channelNo];
}
   
QwtTextLabel * pwrDevice::deviceNameLabel()
{
   return m_deviceNameLabel;
}

void pwrDevice::handleSetProperty( const pcf::IndiProperty & ipRecv )
{
   if(ipRecv.getDevice() != deviceName()) return;
   
   if(ipRecv.getName() == "channelOutlets")
   {
      for(size_t n=0; n < m_numChannels; ++n)
      {
         if( ipRecv.find(m_channels[n]->channelName()))
         {
            std::string outlets =  ipRecv[m_channels[n]->channelName()].get();
            size_t noutlets = std::count(outlets.begin(), outlets.end(), ',');
            std::cerr << "   " << m_channels[n]->channelName() << " " << noutlets+1 << "\n";
            m_channels[n]->numOutlets(noutlets+1);
         }
      }
      
      return;
   }
   
   if(ipRecv.getName() == "channelOnDelays")
   {
      for(size_t n=0; n < m_numChannels; ++n)
      {
         if( ipRecv.find(m_channels[n]->channelName()))
         {
            double onDelay =  ipRecv[m_channels[n]->channelName()].get<double>();
            m_channels[n]->onDelay(onDelay);
         }
      }
      
      return;
   }
   
   if(ipRecv.getName() == "channelOffDelays")
   {
      for(size_t n=0; n < m_numChannels; ++n)
      {
         if( ipRecv.find(m_channels[n]->channelName()))
         {
            double offDelay =  ipRecv[m_channels[n]->channelName()].get<double>();
            m_channels[n]->offDelay(offDelay);
         }
      }
      
      return;
   }
   
   //Check for state
   for(size_t i=0; i< m_numChannels; ++i)
   {
      if( ipRecv.getName() == m_channels[i]->channelName())
      {
         if( ipRecv.find("state") )
         {
            std::string tmp = ipRecv["state"].get();
            
            if(tmp == "On") m_channels[i]->switchState(2);
            if(tmp == "Int") m_channels[i]->switchState(1);
            if(tmp == "Off") m_channels[i]->switchState(0);
         }
      }
   }
   
   if(ipRecv.getName() == "load")
   {
      timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      
      if(ipRecv.find("current"))
      {
         m_current.add( ipRecv["current"].get<double>(), ts);
      }
      
      if(ipRecv.find("voltage"))
      {
         m_voltage.add( ipRecv["voltage"].get<double>(), ts);
      }
      
      if(ipRecv.find("frequency"))
      {
         m_frequency.add( ipRecv["frequency"].get<double>(), ts);
      }
      
      emit loadChanged();
   }
      
}

double pwrDevice::current()
{
   if(m_current.size() == 0) return -1;
   return m_current.lastVal();
}
   
double pwrDevice::voltage()
{
   if(m_voltage.size() == 0) return -1;
   return m_voltage.averageLast(10);
}
   
double pwrDevice::frequency()
{
   if(m_frequency.size() == 0 ) return -1;
   return m_frequency.averageLast(10);
}
   
void pwrDevice::switchOn( const std::string & channelName)
{
   pcf::IndiProperty ip(pcf::IndiProperty::Text);
   
   ip.setDevice(m_deviceName);
   ip.setName(channelName);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = "On";
   
   emit chChange(ip);
}

void pwrDevice::switchOff( const std::string & channelName)
{
   pcf::IndiProperty ip(pcf::IndiProperty::Text);
   
   ip.setDevice(m_deviceName);
   ip.setName(channelName);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = "Off";
   
   emit chChange(ip);

}

} //namespace xqt
