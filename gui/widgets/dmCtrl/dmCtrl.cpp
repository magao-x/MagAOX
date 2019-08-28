
#include "dmCtrl.hpp"


#include <QTimer>


namespace xqt
{
   

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
      if(ipRecv.find("shmimName"))
      {
         m_flatShmim = ipRecv["shmimName"].get<std::string>();
      }
   }
   
   if(ipRecv.getName() == "test")
   {
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
   ui.buttonSetFlat->setEnabled(true);
   ui.buttonZeroFlat->setEnabled(true);
   
   ui.buttonLoadTest->setEnabled(true);
   ui.buttonSetTest->setEnabled(true);
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
   ipFreq["target"] = "flat2.fits";
    
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
