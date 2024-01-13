
#include "pwrGUI.hpp"


#include "qwt_global.h"
#include "qwt_abstract_slider.h"
#include "qwt_abstract_scale_draw.h"
#include "qwt_dial.h"
#include "qwt_dial_needle.h"
#include "qwt_math.h"
#include "qwt_scale_engine.h"
#include "qwt_scale_map.h"
#include "qwt_round_scale_draw.h"
#include "qwt_painter.h"
#include "qwt_text_label.h"

#include <qframe.h>
#include <qpalette.h>
#include <QPaintEvent>
#include <QSlider>

namespace xqt
{

pwrGUI::pwrGUI( QWidget * Parent, Qt::WindowFlags f) : QWidget(Parent, f)
{
   ui.setupUi(this);
   
   ui.totalCurrent->setProperty("isStatus", true);
   ui.averageVoltage->setProperty("isStatus", true);
   ui.averageFrequency->setProperty("isStatus", true);
   
   connect(this, SIGNAL(gotNewDevice(std::string *, std::vector<std::string> *)), this, SLOT(addDevice(std::string *, std::vector<std::string> *)));
   connect(this, SIGNAL(gotDeleteDevice(std::string *)), this, SLOT(removeDevice(std::string *)));

   onDisconnect();
}
   
pwrGUI::~pwrGUI() noexcept
{
   if(m_parent) m_parent->unsubscribe(this);
}

void pwrGUI::pwrGUI::onDisconnect()
{
   QLayoutItem *child;
   while ((child = ui.switchGrid->takeAt(0)) != nullptr) 
   {
      child->widget()->deleteLater(); // delete the widget
      delete child;   // delete the layout item
   }

   for(size_t n=0; n<m_devices.size(); ++n)
   {
      m_devices[n]->deleteLater();
   }
   
   m_devices.clear();

   updateGauges();

   multiIndiSubscriber::onDisconnect();
   
}

void pwrGUI::handleDefProperty( const pcf::IndiProperty & ipRecv /* [in] the property which has changed*/)
{
   bool have = false;
   for(size_t i=0;i<m_devices.size(); ++i)
   {
      if(m_devices[i]->deviceName() == ipRecv.getDevice()) 
      {
         have = true;
         break;
      }
   }
         
   if(!have)
   {
      if(ipRecv.getName() == "channelOutlets")
      {
         size_t nel = ipRecv.getNumElements();
         
         if(nel != 0)
         {
            std::vector<std::string> * elements = new std::vector<std::string>;
            elements->resize(nel);
            for(size_t i = 0; i < nel; ++i)
            {
               (*elements)[i] = ipRecv[i].name();
            }
         
            std::sort(elements->begin(), elements->end());

            std::string * devName = new std::string;
            *devName = ipRecv.getDevice();
         
            emit gotNewDevice( devName, elements);
         }
      }
   }
   
   std::unique_lock<std::mutex> lock(m_addMutex);
   return handleSetProperty(ipRecv);
   
}

void pwrGUI::handleDelProperty( const pcf::IndiProperty & ipRecv /* [in] the property which has deleted*/)
{
   std::string * devName = new std::string;

   *devName = ipRecv.getDevice();

   bool have = false;
   for(size_t i=0;i<m_devices.size(); ++i)
   {
      if(m_devices[i]->deviceName() == *devName) 
      {
         have = true;
         break;
      }
   }

   if(have) emit gotDeleteDevice(devName);
   else delete devName;
}

void pwrGUI::handleSetProperty( const pcf::IndiProperty & ipRecv /* [in] the property which has changed*/)
{
   for(size_t n=0; n<m_devices.size(); ++n)
   {
      if(ipRecv.getDevice() == m_devices[n]->deviceName())
      {
         m_devices[n]->handleSetProperty(ipRecv);
         
         return;
      }
   }
}

void pwrGUI::populateGrid()
{
   eraseGrid();

   std::sort(m_devices.begin(), m_devices.end(), compPwrDevice);

   int currRow = 0;
   for(size_t n =0; n < m_devices.size(); ++n)
   {
      ui.switchGrid->addWidget(m_devices[n]->deviceNameLabel(), currRow, 0, 2, 1);
      m_devices[n]->deviceNameLabel()->show();

      for(size_t i=0;i<m_devices[n]->numChannels();++i)
      {
         ui.switchGrid->addWidget(m_devices[n]->channel(i)->channelNameLabel(), currRow, i+1);
         m_devices[n]->channel(i)->channelNameLabel()->show();
         ui.switchGrid->addWidget(m_devices[n]->channel(i)->channelSwitch(), currRow+1, i+1);
         m_devices[n]->channel(i)->channelSwitch()->show();
      }
      
      currRow +=2;
   }

}
 
void pwrGUI::eraseGrid()
{
   QLayoutItem *child;
   while ((child = ui.switchGrid->takeAt(0)) != nullptr) 
   {
      child->widget()->hide();
      ui.switchGrid->removeWidget(child->widget()); // remove the widget
      delete child;   // delete the layout item
   }
}

void pwrGUI::addDevice( std::string * devName,
                        std::vector<std::string>  * channels
                      )
{
   //First look for it in existing devices
   for(size_t i=0;i<m_devices.size(); ++i)
   {
      if(m_devices[i]->deviceName() == *devName) 
      {
         delete devName;
         delete channels;
         return; //nothing to do
      }
   }
   
   //Get mutex so we don't get clobbered by the next INDI def
   std::unique_lock<std::mutex> lock(m_addMutex);
   
   m_devices.push_back( new pwrDevice(this));
   m_devices.back()->deviceName(*devName);
   m_devices.back()->setChannels(*channels);
   
   QObject::connect(m_devices.back(), SIGNAL(chChange(pcf::IndiProperty &)), this, SLOT(chChange(pcf::IndiProperty &)));
   QObject::connect(m_devices.back(), SIGNAL(loadChanged()), this, SLOT(updateGauges()));
      
   m_parent->addSubscriberProperty(this, m_devices.back()->deviceName(), "load");
   m_parent->addSubscriberProperty(this, m_devices.back()->deviceName(), "channelOutlets");
   m_parent->addSubscriberProperty(this, m_devices.back()->deviceName(), "channelOnDelays");
   m_parent->addSubscriberProperty(this, m_devices.back()->deviceName(), "channelOffDelays");
   
   for(size_t i=0;i<m_devices.back()->numChannels();++i)
   {
      m_parent->addSubscriberProperty(this, m_devices.back()->deviceName(), m_devices.back()->channel(i)->channelName());
   }

   populateGrid();

   delete devName;
   delete channels;
}

void pwrGUI::removeDevice( std::string * devName )
{
   //Get mutex so we don't get clobbered by the next INDI def
   std::unique_lock<std::mutex> lock(m_addMutex);

   //Figure out if we even have to do this
   size_t n;
   for(n = 0; n < m_devices.size(); ++n)
   {
      if( m_devices[n]->deviceName() == *devName) break;
   }
   delete devName;
   
   if(n >= m_devices.size())
   {
      return;
   }

   //First erase grid so there are no active widgets pointing to this, and we're going to redraw it anyway.
   eraseGrid();

   //disconnect signals
   QObject::disconnect(m_devices[n], SIGNAL(chChange(pcf::IndiProperty &)), this, SLOT(chChange(pcf::IndiProperty &)));
   QObject::disconnect(m_devices[n], SIGNAL(loadChanged()), this, SLOT(updateGauges()));

   //unsubscribe
   m_parent->unsubscribe(this, m_devices[n]->deviceName(), "load");
   m_parent->unsubscribe(this, m_devices[n]->deviceName(), "channelOutlets");
   m_parent->unsubscribe(this, m_devices[n]->deviceName(), "channelOnDelays");
   m_parent->unsubscribe(this, m_devices[n]->deviceName(), "channelOffDelays");
   
   for(size_t i=0;i<m_devices[n]->numChannels();++i)
   {
      m_parent->addSubscriberProperty(this, m_devices[n]->deviceName(), m_devices[n]->channel(i)->channelName());
   }

   //Then delete pwrDevice, removing it from m_devices
   pwrDevice * dev = m_devices[n];
   m_devices.erase(m_devices.begin() + n);
   dev->deleteLater();

   //Then populate grid
   populateGrid();

}

void pwrGUI::chChange( pcf::IndiProperty & ip )
{
   try
   {
      sendNewProperty(ip);
   }
   catch(...)
   {
      std::cerr << "pwrGUI::chChange Exception caught\n";
   }
}

void pwrGUI::updateGauges()
{
   double sumCurr = 0, sumVolt = 0, sumFreq = 0;
   size_t nCurr = 0, nVolt =0, nFreq = 0;
   
   for(size_t i =0; i< m_devices.size(); ++i)
   {
      double a = m_devices[i]->current();
      
      if( a >= 0)
      {
         sumCurr += a;
         ++nCurr;
      }
      
      double v = m_devices[i]->voltage(); 
      if( v >= 0)
      {
         sumVolt += v;
         ++nVolt;
      }
      
      double f= m_devices[i]->frequency();
      if( f >= 0)
      {
         sumFreq += f;
         ++nFreq;
      }
      
   }
   
   if(nCurr > 0)
   {
      ui.totalCurrent->display(QString::number(sumCurr, 'f', 1));
   }
   else
   {
      ui.totalCurrent->display("-.-");
   }
   
   if(nVolt > 0)
   {
      ui.averageVoltage->display(QString::number(sumVolt/nVolt, 'f', 1));
   }
   else
   {
      ui.averageVoltage->display("---.-");
   }
   
   if(nFreq > 0)
   {
      ui.averageFrequency->display(QString::number(sumFreq/nFreq, 'f', 1));
   }
   else
   {
      ui.averageFrequency->display("--.-");
   }
}

void pwrGUI::on_buttonReconnect_pressed()
{
   if(m_parent == nullptr) return;
   m_parent->setDisconnect();
}

} //namespace xqt
