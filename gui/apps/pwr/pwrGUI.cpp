
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
   
   connect(this, SIGNAL(gotNewDevice(std::string *, std::vector<std::string> *)), this, SLOT(addNewDevice(std::string *, std::vector<std::string> *)));
   
   n_ACpdus=3;
   currents.resize(n_ACpdus, -1);
   voltages.resize(n_ACpdus, -1);
   frequencies.resize(n_ACpdus, -1);
}
   
pwrGUI::~pwrGUI() noexcept
{
   if(m_parent) m_parent->unsubscribe(this);
   //for(size_t i=0; i< m_devices.size(); ++i) delete m_devices[i];
}

// int pwrGUI::subscribe( multiIndiParent * parent )
// {
//    if(parent == nullptr) return -1;

//    std::cerr << "pwrGUI::subscribe [subscribing to parent]\n";  
//    parent->subscribe(this);
      
//    return 0;
// }

// void pwrGUI::pwrGUI::onConnect()
// {
//    if(m_parent == nullptr) return;
   
//    m_parent->onConnect();
//    //pcf::IndiProperty ipSend;
//    //m_parent->sendGetProperties( ipSend );
// }

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

 
void pwrGUI::handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
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
               (*elements)[i] = ipRecv[i].getName();
            }
         
            std::string * devName = new std::string;
            *devName = ipRecv.getDevice();
         
            emit gotNewDevice( devName, elements);
         }
      }
   }
   
   std::unique_lock<std::mutex> lock(m_addMutex);
   return handleSetProperty(ipRecv);
   
}

void pwrGUI::addNewDevice( std::string * devName,
                           std::vector<std::string>  * channels
                         )
{
   static int currRow = 0;
   
   for(size_t i=0;i<m_devices.size(); ++i)
   {
      if(m_devices[i]->deviceName() == *devName) 
      {
         delete devName;
         delete channels;
         return;
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
   
   
   ui.switchGrid->addWidget(m_devices.back()->deviceNameLabel(), currRow, 0, 2, 1);
   
   for(size_t i=0;i<m_devices.back()->numChannels();++i)
   {
      m_parent->addSubscriberProperty(this, m_devices.back()->deviceName(), m_devices.back()->channel(i)->channelName());
      
      ui.switchGrid->addWidget(m_devices.back()->channel(i)->channelNameLabel(), currRow, i+1);
      ui.switchGrid->addWidget(m_devices.back()->channel(i)->channelSwitch(), currRow+1, i+1);
   }
      
   currRow +=2;
   
   delete devName;
   delete channels;
   
   
}

void pwrGUI::handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
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
 
void pwrGUI::chChange( pcf::IndiProperty & ip )
{
   try
   {
      sendNewProperty(ip);
   }
   catch(...)
   {
      std::cerr << "Exception caught\n";
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

   //multiIndiPublisher * publisher = m_parent;
   //onDisconnect();
   //m_parent->addSubscriber(this);
}

} //namespace xqt
