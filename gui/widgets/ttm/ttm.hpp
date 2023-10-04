
#ifndef ttm_hpp
#define ttm_hpp

#include <QDialog>

#include "ui_ttm.h"

#include "xWidgets/xWidget.hpp"
// #include "../../lib/multiIndiSubscriber.hpp"
// #include "../../lib/multiIndiPublisher.hpp"

namespace xqt 
{
   
class ttm : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_procName;
   
   std::string m_appState;
   
   int m_naxes {2};
   
   double m_pos_1 {0.0};;
   double m_scale_1 {0.01};
   
   double m_pos_2 {0.0};;
   double m_scale_2 {0.01};
   
   double m_pos_3 {0.0};;
   double m_scale_3 {0.01};
   
   std::string m_shmimName;
   bool m_flatSet;
   bool m_testSet;
   
public:
   explicit ttm( std::string & procName,
                 QWidget * Parent = 0, 
                 Qt::WindowFlags f = Qt::WindowFlags()
               );
   
   ~ttm();
   
   void subscribe();

   virtual void onConnect();
   virtual void onDisconnect();
                                   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void sendNewPos1(double np);
   void sendNewPos2(double np);
   void sendNewPos3(double np);
   
public slots:
   void updateGUI();
   
   void on_button_home_pressed();
   void on_button_zero_pressed();
   void on_button_flat_pressed();
   void on_button_release_pressed();

   void on_button_scale_1_pressed();
   void on_button_up_1_pressed();
   void on_button_down_1_pressed();
   
   void on_button_scale_2_pressed();
   void on_button_up_2_pressed();
   void on_button_down_2_pressed();
   
   void on_button_scale_3_pressed();
   void on_button_up_3_pressed();
   void on_button_down_3_pressed();
   
   
private:
     
   Ui::ttm ui;
};
   
ttm::ttm( std::string & procName,
                    QWidget * Parent, 
                    Qt::WindowFlags f) : xWidget(Parent, f), m_procName{procName}
{
   ui.setupUi(this);
   
   setWindowTitle(QString(m_procName.c_str()));
   ui.button_scale_1->setText(QString::number(m_scale_1));
   
   
   
   ui.pos_1->setup(m_procName, "pos_1", statusEntry::FLOAT, "", "");
   ui.pos_1->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.pos_1->format("%0.2f");
   ui.pos_1->onDisconnect();

   ui.pos_2->setup(m_procName, "pos_2", statusEntry::FLOAT, "", "");
   ui.pos_2->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.pos_2->format("%0.2f");
   ui.pos_2->onDisconnect();

   ui.pos_3->setup(m_procName, "pos_3", statusEntry::FLOAT, "", "");
   ui.pos_3->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.pos_3->format("%0.2f");
   ui.pos_3->onDisconnect();

   ui.label_device->setText(m_procName.c_str());
   //ui.label_device_status->setText("unkown");
   ui.label_device_status->device(m_procName);
   ui.label_device_status->setProperty("isStatus", true);

   setXwFont(ui.label_device);
   setXwFont(ui.label_device_status);

   setXwFont(ui.button_home);
   setXwFont(ui.button_flat);
   setXwFont(ui.button_release);
   setXwFont(ui.button_zero);

   setXwFont(ui.label_1);
   setXwFont(ui.label_2);
   setXwFont(ui.label_3);

   
   
   ui.button_scale_1->setProperty("isScaleButton", true);
   ui.button_scale_2->setProperty("isScaleButton", true);
   ui.button_scale_3->setProperty("isScaleButton", true);


   setXwFont(ui.pos_1);
   setXwFont(ui.pos_2);
   setXwFont(ui.pos_3);


   onDisconnect();
}
   

ttm::~ttm()
{
}

void ttm::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "fsm");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "pos_1");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "pos_2");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "pos_3");
   
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "sm_shmimName");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "flat_shmim");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "flat_set");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "test");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "test_shmim");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_procName, "test_set");

   m_parent->addSubscriber(ui.label_device_status);
   m_parent->addSubscriber(ui.pos_1);
   m_parent->addSubscriber(ui.pos_2);
   m_parent->addSubscriber(ui.pos_3);

   return;
}
   
void ttm::onConnect()
{
   ui.label_device_status->onConnect();

   ui.pos_1->onConnect();
   ui.pos_2->onConnect();
   ui.pos_3->onConnect();
}

void ttm::onDisconnect()
{
   ui.label_device_status->onDisconnect();

   ui.pos_1->onDisconnect();
   ui.pos_2->onDisconnect();
   ui.pos_3->onDisconnect();
}

void ttm::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void ttm::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_procName) return;
   
   if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }
   else if(ipRecv.getName() == "pos_1")
   {
      if(ipRecv.find("current"))
      {
         m_pos_1 = ipRecv["current"].get<double>();
      }
   }
   else if(ipRecv.getName() == "pos_2")
   {
      if(ipRecv.find("current"))
      {
         m_pos_2 = ipRecv["current"].get<double>();
      }
   }
   else if(ipRecv.getName() == "pos_3")
   {
      m_naxes = 3;
      if(ipRecv.find("current"))
      {
         m_pos_3 = ipRecv["current"].get<double>();
      }
   }
   else if(ipRecv.getName() == "flat_set")
   {
      if(ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"] == pcf::IndiElement::On) m_flatSet = true;
         else m_flatSet = false;
      }
   }
   else if(ipRecv.getName() == "test_set")
   {
      if(ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"] == pcf::IndiElement::On) m_testSet = true;
         else m_testSet = false;
      }
   }
   else if(ipRecv.getName() == "sm_shmimName")
   {
      if(ipRecv.find("name"))
      {
         m_shmimName = ipRecv["name"].get<std::string>();
      }
   }

   updateGUI();
   
   return;
   
}

void ttm::sendNewPos1(double np)
{
   try
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);

      ipFreq.setDevice(m_procName);
      ipFreq.setName("pos_1");
      ipFreq.add(pcf::IndiElement("current"));
      ipFreq.add(pcf::IndiElement("target"));
      ipFreq["current"] = np;
      ipFreq["target"] = np;
   
      sendNewProperty(ipFreq);   
   }
   catch(...)
   {
      std::cerr << "libcommon INDI exception.  going on. (" << __FILE__ << " " << __LINE__ << "\n";
   }
}

void ttm::sendNewPos2(double np)
{
   try
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);

      ipFreq.setDevice(m_procName);
      ipFreq.setName("pos_2");
      ipFreq.add(pcf::IndiElement("current"));
      ipFreq.add(pcf::IndiElement("target"));
      ipFreq["current"] = np;
      ipFreq["target"] = np;
   
      sendNewProperty(ipFreq);   
   }
   catch(...)
   {
      std::cerr << "libcommon INDI exception.  going on. (" << __FILE__ << " " << __LINE__ << "\n";
   }
}

void ttm::sendNewPos3(double np)
{
   if(m_naxes < 3) return;
   
   try
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);

      ipFreq.setDevice(m_procName);
      ipFreq.setName("pos_3");
      ipFreq.add(pcf::IndiElement("current"));
      ipFreq.add(pcf::IndiElement("target"));
      ipFreq["current"] = np;
      ipFreq["target"] = np;
   
      sendNewProperty(ipFreq);   
   }
   catch(...)
   {
      std::cerr << "libcommon INDI exception.  going on. (" << __FILE__ << " " << __LINE__ << "\n";
   }
}

void ttm::updateGUI()
{
   
   
   if(m_naxes == 3)
   {
      ui.label_1->setText("Piston");
      ui.label_2->setText("Tip");
      ui.label_3->setText("Tilt");
   }
      
   if(m_appState == "NOTHOMED") 
   {
      //ui.label_device_status->setText("RIP");
      
      ui.label_1->setEnabled(false);
      ui.pos_1->setEnabled(false);
      ui.button_up_1->setEnabled(false);
      ui.button_scale_1->setEnabled(false);
      ui.button_down_1->setEnabled(false);
         
      ui.label_2->setEnabled(false);
      ui.pos_2->setEnabled(false);
      ui.button_up_2->setEnabled(false);
      ui.button_scale_2->setEnabled(false);
      ui.button_down_2->setEnabled(false);
      
      ui.label_3->setEnabled(false);
      ui.pos_3->setEnabled(false);
      ui.button_up_3->setEnabled(false);
      ui.button_scale_3->setEnabled(false);
      ui.button_down_3->setEnabled(false);
      
      ui.button_home->setEnabled(true);
      ui.button_zero->setEnabled(false);
      ui.button_flat->setEnabled(false);
      ui.button_release->setEnabled(false);  
   }
   else if(m_appState == "HOMING") 
   {
      //ui.label_device_status->setText("SETTING");
      
      ui.label_1->setEnabled(false);
      ui.pos_1->setEnabled(false);
      ui.button_up_1->setEnabled(false);
      ui.button_scale_1->setEnabled(false);
      ui.button_down_1->setEnabled(false);
         
      ui.label_2->setEnabled(false);
      ui.pos_2->setEnabled(false);
      ui.button_up_2->setEnabled(false);
      ui.button_scale_2->setEnabled(false);
      ui.button_down_2->setEnabled(false);
      
      ui.label_3->setEnabled(false);
      ui.pos_3->setEnabled(false);
      ui.button_up_3->setEnabled(false);
      ui.button_scale_3->setEnabled(false);
      ui.button_down_3->setEnabled(false);
      
      ui.button_home->setEnabled(false);
      ui.button_zero->setEnabled(false);
      ui.button_flat->setEnabled(false);
      ui.button_release->setEnabled(false);  
   }
   else if(m_appState == "READY")
   {
      //ui.label_device_status->setText("SET");
      
      ui.label_1->setEnabled(true);
      ui.pos_1->setEnabled(true);
      ui.button_up_1->setEnabled(true);
      ui.button_scale_1->setEnabled(true);
      ui.button_down_1->setEnabled(true);
         
      ui.label_2->setEnabled(true);
      ui.pos_2->setEnabled(true);
      ui.button_up_2->setEnabled(true);
      ui.button_scale_2->setEnabled(true);
      ui.button_down_2->setEnabled(true);
      
      if(m_naxes == 2)
      {
         ui.label_3->setEnabled(false);
         ui.pos_3->setEnabled(false);
         ui.button_up_3->setEnabled(false);
         ui.button_scale_3->setEnabled(false);
         ui.button_down_3->setEnabled(false);
      }
      else
      {
         ui.label_3->setEnabled(true);
         ui.pos_3->setEnabled(true);
         ui.button_up_3->setEnabled(true);
         ui.button_scale_3->setEnabled(true);
         ui.button_down_3->setEnabled(true);
      }
      
      ui.button_home->setEnabled(false);
      ui.button_zero->setEnabled(true);
      ui.button_flat->setEnabled(true);
      ui.button_release->setEnabled(true);
   }
   else if(m_appState == "OPERATING")
   {
      //ui.label_device_status->setText("OPERATING");
      
      ui.label_1->setEnabled(true);
      ui.pos_1->setEnabled(true);
      ui.button_up_1->setEnabled(true);
      ui.button_scale_1->setEnabled(true);
      ui.button_down_1->setEnabled(true);
         
      ui.label_2->setEnabled(true);
      ui.pos_2->setEnabled(true);
      ui.button_up_2->setEnabled(true);
      ui.button_scale_2->setEnabled(true);
      ui.button_down_2->setEnabled(true);
      
      if(m_naxes == 2)
      {
         ui.label_3->setEnabled(false);
         ui.pos_3->setEnabled(false);
         ui.button_up_3->setEnabled(false);
         ui.button_scale_3->setEnabled(false);
         ui.button_down_3->setEnabled(false);
      }
      else
      {
         ui.label_3->setEnabled(true);
         ui.pos_3->setEnabled(true);
         ui.button_up_3->setEnabled(true);
         ui.button_scale_3->setEnabled(true);
         ui.button_down_3->setEnabled(true);
      }
      
      ui.button_home->setEnabled(false);
      ui.button_zero->setEnabled(true);
      ui.button_flat->setEnabled(true);
      ui.button_release->setEnabled(true);
   }
   else
   {
      //ui.label_device_status->setText(m_appState.c_str());
      
      //Disable & zero all
      ui.label_1->setEnabled(false);
      ui.pos_1->setEnabled(false);
      ui.button_up_1->setEnabled(false);
      ui.button_scale_1->setEnabled(false);
      ui.button_down_1->setEnabled(false);
         
      ui.label_2->setEnabled(false);
      ui.pos_2->setEnabled(false);
      ui.button_up_2->setEnabled(false);
      ui.button_scale_2->setEnabled(false);
      ui.button_down_2->setEnabled(false);
      
      ui.label_3->setEnabled(false);
      ui.pos_3->setEnabled(false);
      ui.button_up_3->setEnabled(false);
      ui.button_scale_3->setEnabled(false);
      ui.button_down_3->setEnabled(false);
      
      ui.button_home->setEnabled(false);
      ui.button_zero->setEnabled(false);
      ui.button_flat->setEnabled(false);
      ui.button_release->setEnabled(false);

      return;
   }

   
} //updateGUI()

void ttm::on_button_home_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_procName);
   ipFreq.setName("initDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void ttm::on_button_zero_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
    ipFreq.setDevice(m_procName);
    ipFreq.setName("flat_set");
    ipFreq.add(pcf::IndiElement("toggle"));
    ipFreq["toggle"] = pcf::IndiElement::Off;
    
    sendNewProperty(ipFreq);
}

void ttm::on_button_flat_pressed()
{
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
    ipFreq.setDevice(m_procName);
    ipFreq.setName("flat_set");
    ipFreq.add(pcf::IndiElement("toggle"));
    ipFreq["toggle"] = pcf::IndiElement::On;
    
    sendNewProperty(ipFreq);
}

void ttm::on_button_release_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_procName);
   ipFreq.setName("releaseDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void ttm::on_button_scale_1_pressed()
{
   if(((int) (100*m_scale_1)) == 1000)
   {
      m_scale_1 = 5.0;
   }
   else if(((int) (100*m_scale_1)) == 500)
   {
      m_scale_1 = 1.0;
   }
   else if(((int) (100*m_scale_1)) == 100)
   {
      m_scale_1 = 0.5;
   }
   else if(((int) (100*m_scale_1)) == 50)
   {
      m_scale_1 = 0.1;
   }
   else if(((int) (100*m_scale_1)) == 10)
   {
      m_scale_1 = 0.05;
   }
   else if(((int) (100*m_scale_1)) == 5)
   {
      m_scale_1 = 0.01;
   }
   else if(((int) (100*m_scale_1)) == 1)
   {
      m_scale_1 = 10.0;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_scale_1);
   ui.button_scale_1->setText(ss);
}

void ttm::on_button_up_1_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_procName);
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos_1 + m_scale_1;
   
   sendNewProperty(ip);
}

void ttm::on_button_down_1_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_procName);
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos_1 - m_scale_1;
   
   sendNewProperty(ip);
}

void ttm::on_button_scale_2_pressed()
{
   if(((int) (100*m_scale_2)) == 1000)
   {
      m_scale_2 = 5.0;
   }
   else if(((int) (100*m_scale_2)) == 500)
   {
      m_scale_2 = 1.0;
   }
   else if(((int) (100*m_scale_2)) == 100)
   {
      m_scale_2 = 0.5;
   }
   else if(((int) (100*m_scale_2)) == 50)
   {
      m_scale_2 = 0.1;
   }
   else if(((int) (100*m_scale_2)) == 10)
   {
      m_scale_2 = 0.05;
   }
   else if(((int) (100*m_scale_2)) == 5)
   {
      m_scale_2 = 0.01;
   }
   else if(((int) (100*m_scale_2)) == 1)
   {
      m_scale_2 = 10.0;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_scale_2);
   ui.button_scale_2->setText(ss);
}

void ttm::on_button_up_2_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_procName);
   ip.setName("pos_2");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos_2 + m_scale_2;
   
   sendNewProperty(ip);
}

void ttm::on_button_down_2_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_procName);
   ip.setName("pos_2");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos_2 - m_scale_2;
   
   sendNewProperty(ip);
}

void ttm::on_button_scale_3_pressed()
{
   if(((int) (100*m_scale_3)) == 1000)
   {
      m_scale_3 = 5.0;
   }
   else if(((int) (100*m_scale_3)) == 500)
   {
      m_scale_3 = 1.0;
   }
   else if(((int) (100*m_scale_3)) == 100)
   {
      m_scale_3 = 0.5;
   }
   else if(((int) (100*m_scale_3)) == 50)
   {
      m_scale_3 = 0.1;
   }
   else if(((int) (100*m_scale_3)) == 10)
   {
      m_scale_3 = 0.05;
   }
   else if(((int) (100*m_scale_3)) == 5)
   {
      m_scale_3 = 0.01;
   }
   else if(((int) (100*m_scale_3)) == 1)
   {
      m_scale_3 = 10.0;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_scale_3);
   ui.button_scale_3->setText(ss);
}

void ttm::on_button_up_3_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_procName);
   ip.setName("pos_3");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos_3 + m_scale_3;
   
   sendNewProperty(ip);
}

void ttm::on_button_down_3_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_procName);
   ip.setName("pos_3");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pos_3 - m_scale_3;
   
   sendNewProperty(ip);
}



} //namespace xqt
   
#include "moc_ttm.cpp"

#endif
