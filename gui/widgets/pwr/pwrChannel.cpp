
#include "pwrChannel.hpp"

#include <iostream>

namespace xqt
{
   
pwrChannel::pwrChannel( QWidget * parent,
                        Qt::WindowFlags flags
                      ) : QWidget(parent, flags)
{
   m_channelNameLabel = new QwtTextLabel(this);
   
   m_channelSwitch = new QSlider(this);
   m_channelSwitch->setOrientation(Qt::Horizontal);
   m_channelSwitch->setMinimum(0);
   m_channelSwitch->setMaximum(10);
   m_channelSwitch->setSingleStep(1);
   m_channelSwitch->setPageStep(1);
   
   QPalette p = m_channelSwitch->palette();
   p.setColor(QPalette::Active, QPalette::Highlight, QColor(22,111,117,255)); // Scale text and line
   p.setColor(QPalette::Inactive, QPalette::Highlight, QColor(22,111,117,255)); // Scale text and line
   m_channelSwitch->setPalette(p);
   
   QObject::connect( m_channelSwitch, SIGNAL(sliderReleased()), this, SLOT(sliderReleased()));
   
   m_timer = new QTimer(this);
   connect(m_timer, SIGNAL(timeout()), this, SLOT(timeOut()));
   connect(this, SIGNAL(switchTargetReached()), this, SLOT(noTimeOut()));
   
}

pwrChannel::~pwrChannel()
{
   if(m_channelNameLabel) delete m_channelNameLabel;
   if(m_channelSwitch) delete m_channelSwitch;
   if(m_timer) delete m_timer;
}

std::string pwrChannel::channelName()
{
   return m_channelName;
}

void pwrChannel::channelName( const std::string & nname)
{
   m_channelName = nname;
   m_channelNameLabel->setText(nname.c_str());
}
   
int pwrChannel::switchState()
{
   if( m_channelSwitch->sliderPosition() > 0.8*(m_channelSwitch->maximum()-m_channelSwitch->minimum()))
   {
      return 2;
   }
   
   return 0;
}
   
void pwrChannel::switchState( int swstate)
{
   if(swstate == 2) 
   {
      m_channelSwitch->setSliderPosition(m_channelSwitch->maximum());
      m_setSwitchState = 2;
      m_channelSwitch->setEnabled(true);
      emit switchTargetReached();
   }
   else if(swstate == 1)
   {
      m_channelSwitch->setSliderPosition(m_channelSwitch->minimum() + 0.5*(m_channelSwitch->maximum()-m_channelSwitch->minimum()));
      m_setSwitchState = 1;
   }
   else 
   {
      m_channelSwitch->setSliderPosition(m_channelSwitch->minimum());
      m_setSwitchState = 0;
      m_channelSwitch->setEnabled(true);
      emit switchTargetReached();
   }
   
   
}   

QwtTextLabel * pwrChannel::channelNameLabel()
{
   return m_channelNameLabel;
}
   
QSlider * pwrChannel::channelSwitch()
{
   return m_channelSwitch;
}

void pwrChannel::numOutlets( int nO)
{
   m_numOutlets = nO;
   calcOnTimeout();
   calcOffTimeout();
}

void pwrChannel::onDelay( double onD )
{
   m_onDelay = onD;
   calcOnTimeout();
}

void pwrChannel::offDelay (double offD)
{
   m_offDelay = offD;
   calcOffTimeout();
}

void pwrChannel::calcOnTimeout()
{
   m_onTimeout = m_numOutlets*10000 + m_onDelay;   
}

void pwrChannel::calcOffTimeout()
{
   m_offTimeout = m_numOutlets*10000 + m_offDelay;
}

void pwrChannel::sliderReleased()
{
   if(m_setSwitchState < 2)
   {
      if( m_channelSwitch->sliderPosition() > m_channelSwitch->minimum()+0.8*(m_channelSwitch->maximum()-m_channelSwitch->minimum()))
      {
         m_channelSwitch->setEnabled(false);
         m_timer->start(m_onTimeout);
         emit switchOn(m_channelName);
      }
      else switchState(0);
   }
   else
   {
      if( m_channelSwitch->sliderPosition() < m_channelSwitch->minimum()+0.2*(m_channelSwitch->maximum()-m_channelSwitch->minimum()))
      {
         m_channelSwitch->setEnabled(false);
         m_timer->start(m_offTimeout);
         emit switchOff(m_channelName);
      }
      else switchState(2);
   }  
}

void pwrChannel::noTimeOut()
{
   m_timer->stop();
}

void pwrChannel::timeOut()
{
   switchState(m_setSwitchState);
}

} //namespace xqt
