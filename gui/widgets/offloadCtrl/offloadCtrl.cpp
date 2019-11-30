
#include "offloadCtrl.hpp"

#include <cmath>
#include <unistd.h>

namespace xqt
{
   
void synchActiveInactive( QPalette * p)
{
   p->setColor(QPalette::Inactive, QPalette::Window, p->color(QPalette::Active, QPalette::Window));
   p->setColor(QPalette::Inactive, QPalette::Background, p->color(QPalette::Active, QPalette::Background));
   p->setColor(QPalette::Inactive, QPalette::WindowText, p->color(QPalette::Active, QPalette::WindowText));
   p->setColor(QPalette::Inactive, QPalette::Foreground, p->color(QPalette::Active, QPalette::Foreground));
   p->setColor(QPalette::Inactive, QPalette::Base, p->color(QPalette::Active, QPalette::Base));
   p->setColor(QPalette::Inactive, QPalette::AlternateBase, p->color(QPalette::Active, QPalette::AlternateBase));
   p->setColor(QPalette::Inactive, QPalette::ToolTipBase, p->color(QPalette::Active, QPalette::ToolTipBase));
   p->setColor(QPalette::Inactive, QPalette::ToolTipText, p->color(QPalette::Active, QPalette::ToolTipText));
   //p->setColor(QPalette::Inactive, QPalette::PlaceholderTetxt, p->color(QPalette::Active, QPalette::PlaceholderTetxt));
   p->setColor(QPalette::Inactive, QPalette::Text, p->color(QPalette::Active, QPalette::Text));
   p->setColor(QPalette::Inactive, QPalette::Button, p->color(QPalette::Active, QPalette::Button));
   p->setColor(QPalette::Inactive, QPalette::ButtonText, p->color(QPalette::Active, QPalette::ButtonText));
   p->setColor(QPalette::Inactive, QPalette::BrightText, p->color(QPalette::Active, QPalette::BrightText));
   p->setColor(QPalette::Inactive, QPalette::Light, p->color(QPalette::Active, QPalette::Light));
   p->setColor(QPalette::Inactive, QPalette::Midlight, p->color(QPalette::Active, QPalette::Midlight));
   p->setColor(QPalette::Inactive, QPalette::Dark, p->color(QPalette::Active, QPalette::Dark));
   p->setColor(QPalette::Inactive, QPalette::Mid, p->color(QPalette::Active, QPalette::Mid));
   p->setColor(QPalette::Inactive, QPalette::Shadow, p->color(QPalette::Active, QPalette::Shadow));
   p->setColor(QPalette::Inactive, QPalette::Highlight, p->color(QPalette::Active, QPalette::Highlight));
   p->setColor(QPalette::Inactive, QPalette::HighlightedText, p->color(QPalette::Active, QPalette::HighlightedText));
   //p->setColor(QPalette::Inactive, QPalette::, p->color(QPalette::Active, QPalette::));
}

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

offloadCtrl::offloadCtrl( QWidget * Parent, Qt::WindowFlags f) : QDialog(Parent, f)
{
//    QPalette p = palette();
//    synchActiveInactive(&p);
//    setPalette(p);
   
   ui.setupUi(this);
   //QPalette p = ui.buttonRest->palette();
   //synchActiveInactive(&p);
   //ui.buttonRest->setPalette(p);
   
    QTimer *timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(updateGUI()));
    timer->start(250);
      
}
   
offloadCtrl::~offloadCtrl()
{
}

int offloadCtrl::subscribe( multiIndiPublisher * publisher )
{
   publisher->subscribeProperty(this, "t2wOffloader", "offload");
   publisher->subscribeProperty(this, "t2wOffloader", "zero");
   publisher->subscribeProperty(this, "t2wOffloader", "gain");
   publisher->subscribeProperty(this, "t2wOffloader", "leak");
   publisher->subscribeProperty(this, "dmtweeter-avg", "nAverage");
   return 0;
}
   
int offloadCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   
   if(ipRecv.getDevice() == "t2wOffloader" || ipRecv.getDevice() == "dmtweeter-avg") 
   {
      return handleSetProperty(ipRecv);
   }
   
   return 0;
}

int offloadCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   //m_mutex.lock();
   if(ipRecv.getDevice() == "t2wOffloader")
   {
      if(ipRecv.getName() == "offload")
      {
         if(ipRecv.find("toggle"))
         {
            if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On) m_t2wOffloadingEnabled = true;
            if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::Off) m_t2wOffloadingEnabled = false;
         }
         
      }
      if(ipRecv.getName() == "gain")
      {
         if(ipRecv.find("current"))
         {
            m_gain = ipRecv["current"].get<double>();
         }
         
      }
      if(ipRecv.getName() == "leak")
      {
         if(ipRecv.find("current"))
         {
            m_leak = ipRecv["current"].get<double>();
         }
         
      }
   }
   
   if(ipRecv.getDevice() == "dmtweeter-avg")
   {
      if(ipRecv.getName() == "nAverage")
      {
         if(ipRecv.find("current"))
         {
            m_navg=ipRecv["current"].get<int>();
            ui.t2w_avg_spin->setValue(m_navg);
         }
      }
   }

   return 0;
   
}

void offloadCtrl::updateGUI()
{

   if(m_t2wOffloadingEnabled)
   {
      ui.t2w_enable->setText("disable");
   }
   else
   {
      ui.t2w_enable->setText("enable");
   }

   char t[8];
   if(!ui.t2w_gain_edit->hasFocus())
   {
      snprintf(t, 8, "%0.2f",m_gain);
      ui.t2w_gain_edit->setText(t);
   }   
   
   if(!ui.t2w_leak_edit->hasFocus())
   {
      snprintf(t, 8, "%0.3f",m_leak);
      ui.t2w_leak_edit->setText(t);
   }   
   
} //updateGUI()

void offloadCtrl::on_t2w_enable_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice("t2wOffloader");
   ip.setName("offload");
   ip.add(pcf::IndiElement("toggle"));
   
   if(m_t2wOffloadingEnabled)
   {
      ip["toggle"].setSwitchState(pcf::IndiElement::Off);
   }
   else
   {
      ip["toggle"].setSwitchState(pcf::IndiElement::On);
   }
   
   sendNewProperty(ip);
}

void offloadCtrl::on_t2w_zero_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice("t2wOffloader");
   ip.setName("zero");
   ip.add(pcf::IndiElement("request"));
   
   if(m_t2wOffloadingEnabled)
   {
      ip["request"].setSwitchState(pcf::IndiElement::Off);
   }
   else
   {
      ip["request"].setSwitchState(pcf::IndiElement::On);
   }
   
   sendNewProperty(ip);
   
}

void offloadCtrl::on_t2w_gain_minus_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("t2wOffloader");
   ip.setName("gain");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_gain - 0.01;
   
   sendNewProperty(ip);
}

void offloadCtrl::on_t2w_gain_plus_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("t2wOffloader");
   ip.setName("gain");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_gain + 0.01;
   
   sendNewProperty(ip);
}

void offloadCtrl::on_t2w_gain_edit_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("t2wOffloader");
   ip.setName("gain");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.t2w_gain_edit->text().toDouble();

   ui.t2w_gain_edit->clearFocus();
   
   sendNewProperty(ip);
   
   
}

void offloadCtrl::on_t2w_leak_minus_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("t2wOffloader");
   ip.setName("leak");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_leak - 0.01;
   
   sendNewProperty(ip);
}

void offloadCtrl::on_t2w_leak_plus_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("t2wOffloader");
   ip.setName("leak");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_leak + 0.01;
   
   sendNewProperty(ip);
}

void offloadCtrl::on_t2w_leak_edit_editingFinished()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("t2wOffloader");
   ip.setName("leak");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.t2w_leak_edit->text().toDouble();

   ui.t2w_leak_edit->clearFocus();
   
   sendNewProperty(ip);
}

void offloadCtrl::on_t2w_avg_spin_valueChanged(int v)
{
   if(v != m_navg)
   {
      pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
      ip.setDevice("dmtweeter-avg");
      ip.setName("nAverage");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = v;

      sendNewProperty(ip);
   }
   
}

} //namespace xqt
