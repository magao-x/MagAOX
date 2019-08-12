
#include "modwfsGUI.hpp"


#include <QTimer>


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

modwfsGUI::modwfsGUI( QWidget * Parent, Qt::WindowFlags f) : QWidget(Parent, f)
{
//    QPalette p = palette();
//    synchActiveInactive(&p);
//    setPalette(p);
   
   ui.setupUi(this);
   QPalette p = ui.buttonRest->palette();
   synchActiveInactive(&p);
   ui.buttonRest->setPalette(p);
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.buttonScale->setText(ss);
   
}
   
modwfsGUI::~modwfsGUI()
{
}

int modwfsGUI::subscribe( multiIndiPublisher * publisher )
{
   publisher->subscribeProperty(this, "modwfs", "state");
   publisher->subscribeProperty(this, "modwfs", "modState");
   publisher->subscribeProperty(this, "modwfs", "modFrequency");
   publisher->subscribeProperty(this, "modwfs", "modRadius");
   publisher->subscribeProperty(this, "fxngenmodwfs", "C1ofst");
   publisher->subscribeProperty(this, "fxngenmodwfs", "C2ofst");
   publisher->subscribeProperty(this, "pdu1", "modttm");  ///\todo this ought to get from the modwfs state code.
   
   return 0;
}
   
int modwfsGUI::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() == "modwfs" || ipRecv.getDevice() == "fxngenmodwfs" || ipRecv.getDevice() == "pdu1")
   {
      return handleSetProperty(ipRecv);
   }
   
   return 0;
}

int modwfsGUI::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() == "modwfs")
   {
      if(ipRecv.getName() == "state")
      {
         if(ipRecv.find("current"))
         {
            m_appState = ipRecv["current"].get<int>();
         }
      }
   
      if(ipRecv.getName() == "modState")
      {
         if(ipRecv.find("current"))
         {
            m_modState = ipRecv["current"].get<int>();
         }
      }
   
      if(ipRecv.getName() == "modFrequency")
      {
         if(ipRecv.find("current"))
         {
            m_modFrequency = ipRecv["current"].get<float>();
         }
      }
   
      if(ipRecv.getName() == "modRadius")
      {
         if(ipRecv.find("current"))
         {
            m_modRadius = ipRecv["current"].get<float>();
         }
      }
   }
   else if(ipRecv.getDevice() == "fxngenmodwfs")
   {
      if(ipRecv.getName() == "C1ofst")
      {
         if(ipRecv.find("value"))
         {
            m_C1ofst = ipRecv["value"].get<double>();
         }
      }
      if(ipRecv.getName() == "C2ofst")
      {
         if(ipRecv.find("value"))
         {
            m_C2ofst = ipRecv["value"].get<double>();
         }
      }
   }
   else if(ipRecv.getDevice() == "pdu1")
   {
      if(ipRecv.getName() == "modttm")
      {
         if(ipRecv.find("state"))
         {
            std::string tmp = ipRecv["state"].get();
            if(tmp == "On") m_pwrState = 2;
            else if(tmp == "Int") m_pwrState = 1;
            else if(tmp == "Off") m_pwrState = 0;
            else m_pwrState = -1;
         }
      }
   }
   
   updateGUI();
   
   //If we get here then we need to add this device
   return 0;
   
}

void modwfsGUI::updateGUI()
{
   
   
   if( m_pwrState != 2 )
   {
      if(m_pwrState == 1)
      {
         ui.ttmStatus->setText("PWR INT");
      }
      else if(m_pwrState == 0)
      {
         ui.ttmStatus->setText("PWR OFF");
      }
      else
      {
         ui.ttmStatus->setText("PWR UNK");
      }
      
      //Disable & zero all
      
      ui.buttonRest->setEnabled(false);
      ui.buttonSet->setEnabled(false);
      ui.buttonModulate->setEnabled(false);
      
      ui.buttonUp->setEnabled(false);
      ui.buttonDown->setEnabled(false);
      ui.buttonLeft->setEnabled(false);
      ui.buttonRight->setEnabled(false);
      ui.buttonScale->setEnabled(false);
      
      ui.voltsAxis1->display(0);
      ui.voltsAxis1->setEnabled(false);
      ui.voltsAxis2->display(0);
      ui.voltsAxis2->setEnabled(false);
      
      return;
   }

   ui.voltsAxis1->display(m_C1ofst);
   ui.voltsAxis2->display(m_C2ofst);
   ui.voltsAxis1->setEnabled(true);
   ui.voltsAxis2->setEnabled(true);

   if( m_modState == 1)
   {
      ui.ttmStatus->setText("REST");
      
      ui.buttonRest->setEnabled(false);
      ui.buttonSet->setEnabled(true);
      ui.buttonModulate->setEnabled(false);
      
      ui.buttonUp->setEnabled(false);
      ui.buttonDown->setEnabled(false);
      ui.buttonLeft->setEnabled(false);
      ui.buttonRight->setEnabled(false);    
      ui.buttonScale->setEnabled(false);
   }
   if( m_modState == 2)
   {
      ui.ttmStatus->setText("SETTING");
      
      ui.buttonRest->setEnabled(true);
      ui.buttonSet->setEnabled(false);
      ui.buttonModulate->setEnabled(false);
      
      ui.buttonUp->setEnabled(false);
      ui.buttonDown->setEnabled(false);
      ui.buttonLeft->setEnabled(false);
      ui.buttonRight->setEnabled(false);
      ui.buttonScale->setEnabled(false);
   }
   if( m_modState == 3)
   {
      ui.ttmStatus->setText("SET");
      
      ui.buttonRest->setEnabled(true);
      ui.buttonSet->setEnabled(false);
      ui.buttonModulate->setEnabled(true);
      
      ui.buttonUp->setEnabled(true);
      ui.buttonDown->setEnabled(true);
      ui.buttonLeft->setEnabled(true);
      ui.buttonRight->setEnabled(true);
      ui.buttonScale->setEnabled(true);
   }
   if( m_modState == 4)
   {
      ui.ttmStatus->setText("MODULATING");
      
      ui.buttonRest->setEnabled(true);
      ui.buttonSet->setEnabled(true);
      ui.buttonModulate->setEnabled(false);
      
      ui.buttonUp->setEnabled(true);
      ui.buttonDown->setEnabled(true);
      ui.buttonLeft->setEnabled(true);
      ui.buttonRight->setEnabled(true);
      ui.buttonScale->setEnabled(true);
   }
   
} //updateGUI()

void modwfsGUI::on_buttonRest_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = 1;
   
   sendNewProperty(ip);
   
}

void modwfsGUI::on_buttonSet_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = 3;
   
   sendNewProperty(ip);
   
}

void modwfsGUI::on_buttonModulate_pressed()
{
   pcf::IndiProperty ipRad(pcf::IndiProperty::Number);
   
   ipRad.setDevice("modwfs");
   ipRad.setName("modRadius");
   ipRad.add(pcf::IndiElement("requested"));
   ipRad["requested"] = 6.0;
    
   sendNewProperty(ipRad);
   
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);
   
   ipFreq.setDevice("modwfs");
   ipFreq.setName("modFrequency");
   ipFreq.add(pcf::IndiElement("requested"));
   ipFreq["requested"] = 1000.0;
    
   sendNewProperty(ipFreq);
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = 4;
   
   sendNewProperty(ip);
   
}

void modwfsGUI::on_buttonUp_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fxngenmodwfs");
   ip.setName("C1ofst");
   ip.add(pcf::IndiElement("value"));
   ip["value"] = m_C1ofst + m_stepSize;
    
   sendNewProperty(ip);
   
}

void modwfsGUI::on_buttonDown_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fxngenmodwfs");
   ip.setName("C1ofst");
   ip.add(pcf::IndiElement("value"));
   ip["value"] = m_C1ofst - m_stepSize;
    
   sendNewProperty(ip);
   
}

void modwfsGUI::on_buttonLeft_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fxngenmodwfs");
   ip.setName("C2ofst");
   ip.add(pcf::IndiElement("value"));
   ip["value"] = m_C2ofst - m_stepSize;
    
   sendNewProperty(ip);
   
}

void modwfsGUI::on_buttonRight_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fxngenmodwfs");
   ip.setName("C2ofst");
   ip.add(pcf::IndiElement("value"));
   ip["value"] = m_C2ofst + m_stepSize;
    
   sendNewProperty(ip);
   

}

void modwfsGUI::on_buttonScale_pressed()
{
   if(((int) (100*m_stepSize)) == 1)
   {
      m_stepSize = 0.05;
   }
   else if(((int) (100*m_stepSize)) == 5)
   {
      m_stepSize = 0.1;
   }
   else if(((int) (100*m_stepSize)) == 10)
   {
      m_stepSize = 0.01;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.buttonScale->setText(ss);


}

} //namespace xqt
