
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

void outputRGBa(const QColor & qc)
{
   int r, g, b, a;
   qc.getRgb(&r,&g, &b, &a);
   
   std::cout << r << "," << g << "," << b << "," << a;
}


void paletteExporter( QWidget * w )
{
   QPalette p = w->palette();
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Window, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Window));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Background, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Background));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::WindowText, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::WindowText));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Foreground, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Foreground));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Base, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Base));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::AlternateBase, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::AlternateBase));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::ToolTipBase, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::ToolTipBase));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::ToolTipText, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::ToolTipText));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Text, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Text));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Button, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Button));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::ButtonText, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::ButtonText));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::BrightText, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::BrightText));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Light, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Light));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Midlight, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Midlight));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Dark, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Dark));
   std::cout << "));\n";
   
   std::cout << "   p->setColor(QPalette::Active, QPalette::Shadow, QColor(";
   outputRGBa(p.color(QPalette::Active, QPalette::Shadow));
   std::cout << "));\n";
   
};

namespace xqt
{
   
void setXDialPalette( QPalette * p)
{
   p->setColor(QPalette::Active, QPalette::Foreground, QColor(0,0,0,0));
   p->setColor(QPalette::Active, QPalette::Base, QColor(0,0,0,0));
   p->setColor(QPalette::Active, QPalette::Text, QColor(22,111,117,255));
   p->setColor(QPalette::Active, QPalette::BrightText, QColor(113,0,0,255)); // Scale text and line
   
   p->setColor(QPalette::Inactive, QPalette::Foreground, QColor(0,0,0,0));//Dial circle color   
   p->setColor(QPalette::Inactive, QPalette::Base, QColor(0,0,0,0));//Overall Background
   p->setColor(QPalette::Inactive, QPalette::Text, QColor(22,111,117,255)); // Scale text and line
   p->setColor(QPalette::Inactive, QPalette::BrightText, QColor(113,0,0,255)); // Scale text and line
   
   p->setColor(QPalette::Disabled, QPalette::Foreground, QColor(0,0,0,0));//Dial circle color   
   p->setColor(QPalette::Disabled, QPalette::Base, QColor(0,0,0,0));//Overall Background
   p->setColor(QPalette::Disabled, QPalette::Text, QColor(22,111,117,255)); // Scale text and line
   p->setColor(QPalette::Disabled, QPalette::BrightText, QColor(113,0,0,255)); // Scale text and line
   
};

pwrGUI::pwrGUI( QWidget * Parent, Qt::WindowFlags f) : QWidget(Parent, f)
{
   ui.setupUi(this);
   
//    pdu1Needle = new QwtDialSimpleNeedle(QwtDialSimpleNeedle::Style::Arrow,false, Qt::red);
//    ui.currentDial->setNeedle(pdu1Needle);
   
   ui.currentDial->setOrigin(150.0);
   ui.currentDial->setScaleArc(0.0,240.0);
   ui.currentDial->setScale(0.0,15.0);
   ui.currentDial->setScaleStepSize(1.0);
   ui.currentDial->setScaleMaxMajor(5);
   ui.currentDial->setScaleMaxMinor(1);
   
   //paletteExporter(ui.voltageDial);
   
   QPalette p = ui.currentDial->palette();
   setXDialPalette(&p);
   ui.currentDial->setPalette(p);
   
   ui.currentDial->setNumNeedles(2);
   ui.currentDial->setValue(0, 0.0);
   ui.currentDial->setValue(1, 0.0);
   ui.currentDial->setFocusPolicy(Qt::NoFocus);
   
   ui.voltageDial->setOrigin(150.0);
   ui.voltageDial->setScaleArc(0.0,240.0);
   ui.voltageDial->setScale(118.0,122.0);
   ui.voltageDial->setScaleStepSize(1.0);
   ui.voltageDial->setScaleMaxMajor(10);
   ui.voltageDial->setScaleMaxMinor(10);
   
   p = ui.voltageDial->palette();
   setXDialPalette(&p);
   ui.voltageDial->setPalette(p);
   
   ui.voltageDial->setNumNeedles(2);
   ui.voltageDial->setValue(0, 0.0);
   ui.voltageDial->setValue(1, 0.0);
   ui.voltageDial->setFocusPolicy(Qt::NoFocus);
   
   ui.frequencyDial->setOrigin(150.0);
   ui.frequencyDial->setScaleArc(0.0,240.0);
   ui.frequencyDial->setScale(58.0,62.0);
   ui.frequencyDial->setScaleStepSize(1.0);
   ui.frequencyDial->setScaleMaxMajor(10);
   ui.frequencyDial->setScaleMaxMinor(10);
   
   p = ui.frequencyDial->palette();
   setXDialPalette(&p);
   ui.frequencyDial->setPalette(p);
   
   ui.frequencyDial->setNumNeedles(2);
   ui.frequencyDial->setValue(0, 0.0);
   ui.frequencyDial->setValue(1, 0.0);
   ui.frequencyDial->setFocusPolicy(Qt::NoFocus);
   
   m_devices.resize(2);
   m_devices[0] = new pwrDevice(this);
   m_devices[1] = new pwrDevice(this);
   
   m_devices[0]->deviceName("pdu0");
   m_devices[0]->setChannels({"psdcmain", "swint", "safety", "compicc", "comprtc", "comppcie1", "modttm"});
   
   QObject::connect(m_devices[0], SIGNAL(chChange(pcf::IndiProperty &)), this, SLOT(chChange(pcf::IndiProperty &)));
   QObject::connect(m_devices[0], SIGNAL(loadChanged()), this, SLOT(updateGauges()));
   

   ui.switchGrid->addWidget(m_devices[0]->deviceNameLabel(), 0, 0, 2, 1);
   
   for(size_t n=0; n<m_devices[0]->numChannels();++n)
   {
      ui.switchGrid->addWidget(m_devices[0]->channel(n)->channelNameLabel(), 0, n+1);
      ui.switchGrid->addWidget(m_devices[0]->channel(n)->channelSwitch(), 1, n+1);
   }
   
   
   QObject::connect(m_devices[1], SIGNAL(chChange(pcf::IndiProperty &)), this, SLOT(chChange(pcf::IndiProperty &)));
   QObject::connect(m_devices[1], SIGNAL(loadChanged()), this, SLOT(updateGauges()));
   
   m_devices[1]->deviceName("pdu1");
   m_devices[1]->setChannels({"camwfs", "dmwoofer", "camlowfs", "dmcoron", "camsci1", "camsci2", "zaberstages"});
   
   ui.switchGrid->addWidget(m_devices[1]->deviceNameLabel(), 2, 0,2,1);
   for(size_t n=0; n<m_devices[1]->numChannels();++n)
   {
      ui.switchGrid->addWidget(m_devices[1]->channel(n)->channelNameLabel(), 2, n+1);
      ui.switchGrid->addWidget(m_devices[1]->channel(n)->channelSwitch(), 3, n+1);
   }
   
   n_ACpdus=2;
   currents.resize(n_ACpdus, -1);
   voltages.resize(n_ACpdus, -1);
   frequencies.resize(n_ACpdus, -1);
}
   
pwrGUI::~pwrGUI()
{
}

int pwrGUI::subscribe( multiIndiPublisher * publisher )
{
   for(size_t n=0; n<m_devices.size(); ++n)
   {
      publisher->subscribeProperty(this, m_devices[n]->deviceName(), "load");
      publisher->subscribeProperty(this, m_devices[n]->deviceName(), "channelOutlets");
      publisher->subscribeProperty(this, m_devices[n]->deviceName(), "channelOnDelays");
      publisher->subscribeProperty(this, m_devices[n]->deviceName(), "channelOffDelays");
      for(size_t i=0;i<m_devices[n]->numChannels();++i)
      {
         publisher->subscribeProperty(this, m_devices[n]->deviceName(), m_devices[n]->channel(i)->channelName());
      }
   }
      
   return 0;
}
                                
int pwrGUI::handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/)
{
   std::string key = ipRecv.createUniqueKey();
   
   for(size_t n=0; n<m_devices.size(); ++n)
   {
      if(ipRecv.getDevice() == m_devices[n]->deviceName())
      {
         m_devices[n]->handleSetProperty(ipRecv);
         
         //updateAverages();
         return 0;
      }
   }

   //If we get here then we need to add this device
   return 0;
   
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
         ui.currentDial->setValue(i,a);
         sumCurr += a;
         ++nCurr;
      }
      
      double v = m_devices[i]->voltage(); 
      if( v >= 0)
      {
         ui.voltageDial->setValue(i,v);
         sumVolt += v;
         ++nVolt;
      }
      
      double f= m_devices[i]->frequency();
      if( f >= 0)
      {
         ui.frequencyDial->setValue(i,f);
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

} //namespace xqt
