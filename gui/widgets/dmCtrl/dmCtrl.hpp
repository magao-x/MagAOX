
#ifndef dmCtrl_hpp
#define dmCtrl_hpp

#include <QDialog>

#include "ui_dmCtrl.h"

#include "../../lib/multiIndi.hpp"

namespace xqt 
{
   
class dmCtrl : public QDialog, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   std::string m_dmName;
   std::string m_shmimName;
   
   std::string m_flatShmim;
   std::string m_flatName;
   std::string m_flatTarget;
   
   std::string m_testShmim;
   std::string m_testName;
   std::string m_testTarget;
   
public:
   dmCtrl( std::string & dmName,
           QWidget * Parent = 0, 
           Qt::WindowFlags f = 0
         );
   
   ~dmCtrl();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
public slots:
   void updateGUI();
   
   void on_buttonInit_pressed();
   void on_buttonZero_pressed();
   void on_buttonRelease_pressed();
   
   void on_buttonLoadFlat_pressed();
   void on_buttonSetFlat_pressed();
   void on_buttonZeroFlat_pressed();
   
   void on_buttonLoadTest_pressed();
   void on_buttonSetTest_pressed();
   void on_buttonZeroTest_pressed();
   
private:
     
   Ui::dmCtrl ui;
};
   
dmCtrl::dmCtrl( std::string & dmName,
                QWidget * Parent, 
                Qt::WindowFlags f) : QDialog(Parent, f), m_dmName{dmName}
{
   ui.setupUi(this);
   ui.labelDMName->setText(m_dmName.c_str());
   
   setWindowTitle(QString(m_dmName.c_str()));
}
   
dmCtrl::~dmCtrl()
{
}

int dmCtrl::subscribe( multiIndiPublisher * publisher )
{
   if(!publisher) return -1;
   
   publisher->subscribeProperty(this, m_dmName, "fsm");
   publisher->subscribeProperty(this, m_dmName, "sm_shmimName");
   publisher->subscribeProperty(this, m_dmName, "flat");
   publisher->subscribeProperty(this, m_dmName, "test");
//   publisher->subscribeProperty(this, m_dmName, "");
//   publisher->subscribeProperty(this, m_dmName, "");
   
   return 0;
}
   
int dmCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
   
   return 0;
}

int dmCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_dmName) return 0;
   
   if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }
   
   if(ipRecv.getName() == "sm_shmimName")
   {
      if(ipRecv.find("name"))
      {
         m_shmimName = ipRecv["name"].get<std::string>();
      }
   }
   
   if(ipRecv.getName() == "flat")
   {
      if(ipRecv.find("current"))
      {
         m_flatName = ipRecv["current"].get<std::string>();
      }
      
      if(ipRecv.find("target"))
      {
         m_flatTarget = ipRecv["target"].get<std::string>();
      }
      
      if(ipRecv.find("shmimName"))
      {
         m_flatShmim = ipRecv["shmimName"].get<std::string>();
      }
   }
   
   if(ipRecv.getName() == "test")
   {
      if(ipRecv.find("current"))
      {
         m_testName = ipRecv["current"].get<std::string>();
      }
      
      if(ipRecv.find("target"))
      {
         m_testTarget = ipRecv["target"].get<std::string>();
      }
      
      if(ipRecv.find("shmimName"))
      {
         m_testShmim = ipRecv["shmimName"].get<std::string>();
      }
   }
   
   updateGUI();
   
   //If we get here then we need to add this device
   return 0;
   
}

void dmCtrl::updateGUI()
{
   ui.dmStatus->setText(m_appState.c_str());
   ui.labelShmimName_value->setText(m_shmimName.c_str());
   ui.labelFlatShmim_value->setText(m_flatShmim.c_str());
   ui.labelTestShmim_value->setText(m_testShmim.c_str());

   
   if( m_appState != "READY" && m_appState != "OPERATING" )
   {
      //Disable & zero all
      
      ui.buttonInit->setEnabled(false);
      ui.buttonZero->setEnabled(false);
      ui.buttonRelease->setEnabled(false);
      ui.buttonLoadFlat->setEnabled(true);
      ui.buttonSetFlat->setEnabled(false);
      ui.buttonZeroFlat->setEnabled(false);
      
      ui.buttonLoadTest->setEnabled(true);
      ui.buttonSetTest->setEnabled(false);
      ui.buttonZeroTest->setEnabled(false);
      
      return;
   }
   
   if( m_appState == "READY" )
   {
      
      ui.buttonInit->setEnabled(true);
      ui.buttonZero->setEnabled(false);
      ui.buttonRelease->setEnabled(false);
      ui.buttonLoadFlat->setEnabled(true);
      ui.buttonSetFlat->setEnabled(false);
      ui.buttonZeroFlat->setEnabled(false);
      
      ui.buttonLoadTest->setEnabled(true);
      ui.buttonSetTest->setEnabled(false);
      ui.buttonZeroTest->setEnabled(false);
      
      return;
   }
   
   ui.buttonInit->setEnabled(false);
   ui.buttonZero->setEnabled(true);
   ui.buttonRelease->setEnabled(true);
   ui.buttonLoadFlat->setEnabled(true);
   

   if(m_flatName == m_flatTarget && m_flatName != "")
   {
      ui.buttonSetFlat->setEnabled(true);   
   }
   else
   {
      ui.buttonSetFlat->setEnabled(false);
   }   
   ui.buttonZeroFlat->setEnabled(true);

   if(m_testName == m_testTarget && m_testName != "")
   {
      ui.buttonSetTest->setEnabled(true);   
   }
   else
   {
      ui.buttonSetTest->setEnabled(false);
   }
   ui.buttonZeroTest->setEnabled(true);
      
} //updateGUI()

void dmCtrl::on_buttonInit_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("initDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);   
}

void dmCtrl::on_buttonZero_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("zeroDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonRelease_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("releaseDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonLoadFlat_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("flat");
   ipFreq.add(pcf::IndiElement("target"));
   ipFreq["target"] = "flat.fits";
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonSetFlat_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("setFlat");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonZeroFlat_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("zeroFlat");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonLoadTest_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("test");
   ipFreq.add(pcf::IndiElement("target"));
   ipFreq["target"] = "test.fits";
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonSetTest_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("setTest");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonZeroTest_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Text);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("zeroTest");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"] = 1;
    
   sendNewProperty(ipFreq);
}

} //namespace xqt
   
#include "moc_dmCtrl.cpp"

#endif
