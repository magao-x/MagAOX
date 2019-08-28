
#include "ttmpupilGUI.hpp"


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

ttmpupilGUI::ttmpupilGUI( QWidget * Parent, Qt::WindowFlags f) : QWidget(Parent, f)
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
   
ttmpupilGUI::~ttmpupilGUI()
{
}

int ttmpupilGUI::subscribe( multiIndiPublisher * publisher )
{
   publisher->subscribeProperty(this, "ttmpupil", "fsm");
   publisher->subscribeProperty(this, "ttmpupil", "pos_1");
   publisher->subscribeProperty(this, "ttmpupil", "pos_2");
   
   return 0;
}
   
int ttmpupilGUI::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() == "ttmpupil") 
   {
      return handleSetProperty(ipRecv);
   }
   
   return 0;
}

int ttmpupilGUI::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() == "ttmpupil")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_appState = ipRecv["state"].get<std::string>();
         }
      }
   
      if(ipRecv.getName() == "pos_1")
      {
         if(ipRecv.find("current"))
         {
            m_pos1= ipRecv["current"].get<double>();
         }
      }
   
      if(ipRecv.getName() == "pos_2")
      {
         if(ipRecv.find("current"))
         {
            m_pos2 = ipRecv["current"].get<double>();
         }
      }
   
   
      updateGUI();
   }
   
   return 0;
   
}

void ttmpupilGUI::updateGUI()
{
   
   
   if( m_appState != "READY" && m_appState != "OPERATING" )
   {
      //Disable & zero all
      
      ui.buttonRest->setEnabled(false);
      ui.buttonSet->setEnabled(false);
      
      ui.buttonUp->setEnabled(false);
      ui.buttonDown->setEnabled(false);
      ui.buttonLeft->setEnabled(false);
      ui.buttonRight->setEnabled(false);
      ui.buttonScale->setEnabled(false);
      
      ui.voltsAxis1->display(0);
      ui.voltsAxis1->setEnabled(false);
      ui.voltsAxis2->display(0);
      ui.voltsAxis2->setEnabled(false);

      
      if(m_appState == "NOTHOMED")
      {
         ui.buttonSet->setEnabled(true);
         ui.ttmStatus->setText("RESTED");
      }
      else if(m_appState == "HOMING")
      {
         ui.ttmStatus->setText("SETTING");
      }
      else 
      {
         ui.ttmStatus->setText(m_appState.c_str());
      }
      
      
      return;
   }

   ui.voltsAxis1->display(m_pos1);
   ui.voltsAxis2->display(m_pos2);
   ui.voltsAxis1->setEnabled(true);
   ui.voltsAxis2->setEnabled(true);

  
   if(m_appState == "READY"  )
   {
      ui.ttmStatus->setText("SET");
      
      ui.buttonRest->setEnabled(true);
      ui.buttonSet->setEnabled(false);
      
      ui.buttonUp->setEnabled(true);
      ui.buttonDown->setEnabled(true);
      ui.buttonLeft->setEnabled(true);
      ui.buttonRight->setEnabled(true);
      ui.buttonScale->setEnabled(true);
   }
   if(m_appState == "OPERATING")
   {
      ui.ttmStatus->setText("CLOSED LOOP");
      
      ui.buttonRest->setEnabled(true);
      ui.buttonSet->setEnabled(true);
      
      ui.buttonUp->setEnabled(false);
      ui.buttonDown->setEnabled(false);
      ui.buttonLeft->setEnabled(false);
      ui.buttonRight->setEnabled(false);
      ui.buttonScale->setEnabled(false);
   }
   
} //updateGUI()

void ttmpupilGUI::on_buttonRest_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("releaseDM");
   ip.add(pcf::IndiElement("request"));
   ip["request"] = 1;
   
   sendNewProperty(ip);
   
}

void ttmpupilGUI::on_buttonSet_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("initDM");
   ip.add(pcf::IndiElement("request"));
   ip["request"] = 1;
   
   sendNewProperty(ip);
   
}


void ttmpupilGUI::on_buttonUp_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos1 + m_stepSize;
    
   sendNewProperty(ip);
   
}

void ttmpupilGUI::on_buttonDown_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos1 - m_stepSize;
    
   sendNewProperty(ip);
   
}

void ttmpupilGUI::on_buttonLeft_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_2");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos2 - m_stepSize;
    
   sendNewProperty(ip);
   
}

void ttmpupilGUI::on_buttonRight_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_2");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos2 + m_stepSize;
    
   sendNewProperty(ip);
   

}

void ttmpupilGUI::on_buttonScale_pressed()
{
   if(((int) (100*m_stepSize)) == 10)
   {
      m_stepSize = 0.5;
   }
   else if(((int) (100*m_stepSize)) == 50)
   {
      m_stepSize = 1.0;
   }
   else if(((int) (100*m_stepSize)) == 100)
   {
      m_stepSize = 0.1;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.buttonScale->setText(ss);


}

} //namespace xqt
