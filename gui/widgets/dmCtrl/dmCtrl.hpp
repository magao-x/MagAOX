
#ifndef dmCtrl_hpp
#define dmCtrl_hpp

#include "ui_dmCtrl.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt 
{
   
class dmCtrl : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   std::string m_dmName;
   std::string m_shmimName;
   
   std::string m_flatShmim;
   bool m_flatSet {false};
   std::string m_flatName;
   
   std::string m_testShmim;
   bool m_testSet {false};
   std::string m_testName;

   
public:
   explicit dmCtrl( std::string & dmName,
                    QWidget * Parent = 0, 
                    Qt::WindowFlags f = Qt::WindowFlags()
                  );
   
   ~dmCtrl();
   
   void subscribe();
             
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
public slots:
   void updateGUI();
   
   void on_buttonInit_pressed();
   void on_buttonZeroAll_pressed();
   void on_buttonZero_pressed();
   void on_buttonRelease_pressed();
   
   void on_comboSelectFlat_activated(int);
   void on_buttonSetFlat_pressed();
   void on_buttonZeroFlat_pressed();

   void on_comboSelectTest_activated(int);
   void on_buttonSetTest_pressed();
   void on_buttonZeroTest_pressed();
   
signals: 

   void doUpdateGUI();

private:
     
   Ui::dmCtrl ui;
};
   
dmCtrl::dmCtrl( std::string & dmName,
                QWidget * Parent, 
                Qt::WindowFlags f) : xWidget(Parent, f), m_dmName{dmName}
{
   ui.setupUi(this);
   //ui.labelDMName->setText(m_dmName.c_str());
   
   setWindowTitle(QString(m_dmName.c_str()));

   ui.fsmState->device(m_dmName);

   setXwFont(ui.buttonInit);
   setXwFont(ui.buttonZeroAll);
   setXwFont(ui.buttonZero);
   setXwFont(ui.buttonInit);  
   setXwFont(ui.buttonRelease);
   setXwFont(ui.buttonSetFlat);
   setXwFont(ui.buttonZeroFlat);
   setXwFont(ui.buttonSetTest);
   setXwFont(ui.buttonZeroTest);
   setXwFont(ui.comboSelectFlat);
   setXwFont(ui.comboSelectTest);
   
   //setXwFont(ui.fsmState);
   
   setXwFont(ui.labelShmimName);
   setXwFont(ui.labelShmimName_value);
   setXwFont(ui.labelFlatShmim);
   setXwFont(ui.labelFlatShmim_value);
   setXwFont(ui.labelTestShmim);
   setXwFont(ui.labelTestShmim_value);
   
   connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

   onDisconnect();
}
   
dmCtrl::~dmCtrl()
{
   if(m_parent) m_parent->unsubscribe(this);
}

void dmCtrl::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty(this, m_dmName, "fsm");
   m_parent->addSubscriberProperty(this, m_dmName, "sm_shmimName");
   m_parent->addSubscriberProperty(this, m_dmName, "flat");
   m_parent->addSubscriberProperty(this, m_dmName, "flat_shmim");
   m_parent->addSubscriberProperty(this, m_dmName, "flat_set");
   m_parent->addSubscriberProperty(this, m_dmName, "test");
   m_parent->addSubscriberProperty(this, m_dmName, "test_shmim");
   m_parent->addSubscriberProperty(this, m_dmName, "test_set");
   m_parent->addSubscriber(ui.fsmState);

   return;
}
  
void dmCtrl::onConnect()
{
   //ui.labelDMName->setEnabled(true);
   ui.fsmState->setEnabled(true);
   ui.labelShmimName->setEnabled(true);
   ui.labelShmimName_value->setEnabled(true);
   ui.labelFlatShmim->setEnabled(true);
   ui.labelFlatShmim_value->setEnabled(true);
   ui.labelTestShmim->setEnabled(true);
   ui.labelTestShmim_value->setEnabled(true);

   ui.buttonZeroAll->setEnabled(true);
      
   ui.comboSelectFlat->setEnabled(true);
   ui.comboSelectTest->setEnabled(true);
   
   ui.fsmState->onConnect();

   setWindowTitle(QString(m_dmName.c_str()));
}

void dmCtrl::onDisconnect()
{
   //ui.labelDMName->setEnabled(false);
   ui.fsmState->setEnabled(false);
   ui.labelShmimName->setEnabled(false);
   ui.labelShmimName_value->setEnabled(false);
   ui.labelFlatShmim->setEnabled(false);
   ui.labelFlatShmim_value->setEnabled(false);
   ui.labelTestShmim->setEnabled(false);
   ui.labelTestShmim_value->setEnabled(false);
   
   ui.buttonInit->setEnabled(false);
   ui.buttonZero->setEnabled(false);
   ui.buttonZeroAll->setEnabled(false);
   ui.buttonRelease->setEnabled(false);

   ui.buttonSetFlat->setEnabled(false);
   ui.buttonZeroFlat->setEnabled(false);
   
   ui.buttonSetTest->setEnabled(false);
   ui.buttonZeroTest->setEnabled(false);
      
   ui.comboSelectFlat->setEnabled(false);
   ui.comboSelectTest->setEnabled(false);
   
   setWindowTitle(QString(m_dmName.c_str()) + QString(" (disconnected)"));

   ui.fsmState->onDisconnect();
   
   multiIndiSubscriber::onDisconnect();
}

void dmCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void dmCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_dmName) 
   {  
      return;
   }
   else if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }
   else if(ipRecv.getName() == "sm_shmimName")
   {
      if(ipRecv.find("name"))
      {
         m_shmimName = ipRecv["name"].get<std::string>();
      }
   }
   else if(ipRecv.getName() == "flat")
   {
      ui.comboSelectFlat->clear();
      
      for(auto it=ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it)
      {
         if(ui.comboSelectFlat->findText(it->first.c_str(), Qt::MatchExactly | Qt::MatchCaseSensitive) == -1)
         {
            ui.comboSelectFlat->addItem(it->first.c_str());
         }
         
         if(ipRecv[it->first] == pcf::IndiElement::On) ui.comboSelectFlat->setCurrentText(it->first.c_str()); 
      }
   }
   else if(ipRecv.getName() == "flat_shmim")
   {
      if(ipRecv.find("channel"))
      {
         m_flatShmim = ipRecv["channel"].get<std::string>();
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
   else if(ipRecv.getName() == "test")
   {
      ui.comboSelectTest->clear();
      
      for(auto it=ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it)
      {
         if(ui.comboSelectTest->findText(it->first.c_str(), Qt::MatchExactly | Qt::MatchCaseSensitive) == -1)
         {
            ui.comboSelectTest->addItem(it->first.c_str());
         }
         
         if(ipRecv[it->first] == pcf::IndiElement::On) ui.comboSelectTest->setCurrentText(it->first.c_str());
         
      }
   }
   else if(ipRecv.getName() == "test_shmim")
   {
      if(ipRecv.find("channel"))
      {
         m_testShmim = ipRecv["channel"].get<std::string>();
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
   
   emit doUpdateGUI();
   
}

void dmCtrl::updateGUI()
{
   ui.labelShmimName_value->setText(m_shmimName.c_str());
   ui.labelFlatShmim_value->setText(m_flatShmim.c_str());
   ui.labelTestShmim_value->setText(m_testShmim.c_str());

//    ui.buttonSetFlat->setEnabled(true);
//    ui.buttonZeroFlat->setEnabled(true);
//    
//    ui.buttonSetTest->setEnabled(true);
//    ui.buttonZeroTest->setEnabled(true);
//       
   if( m_appState != "READY" && m_appState != "OPERATING" )
   {
      //Disable & zero all
      
      ui.buttonInit->setEnabled(false);
      ui.buttonZero->setEnabled(false);
      ui.buttonRelease->setEnabled(false);

      ui.buttonSetFlat->setEnabled(false);
      ui.buttonZeroFlat->setEnabled(false);
      
      ui.buttonSetTest->setEnabled(false);
      ui.buttonZeroTest->setEnabled(false);
      
      return;
   }
   
   if( m_appState == "READY" )
   {
      
      ui.buttonInit->setEnabled(true);
      ui.buttonZero->setEnabled(false);
      ui.buttonRelease->setEnabled(false);

      ui.buttonSetFlat->setEnabled(false);
      ui.buttonZeroFlat->setEnabled(false);
      
      ui.buttonSetTest->setEnabled(false);
      ui.buttonZeroTest->setEnabled(false);
      
      return;
   }
   
   ui.buttonInit->setEnabled(false);
   ui.buttonZero->setEnabled(true);
   ui.buttonRelease->setEnabled(true);
   

   if(m_flatSet == false)
   {
      ui.buttonSetFlat->setEnabled(true);   
      ui.buttonZeroFlat->setEnabled(false);
   }
   else
   {
      ui.buttonSetFlat->setEnabled(false);
      ui.buttonZeroFlat->setEnabled(true);
   }

   if(m_testSet == false)
   {
      ui.buttonSetTest->setEnabled(true);   
      ui.buttonZeroTest->setEnabled(false);
   }
   else
   {
      ui.buttonSetTest->setEnabled(false);
      ui.buttonZeroTest->setEnabled(true);
   }

} //updateGUI()

void dmCtrl::on_buttonInit_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("initDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void dmCtrl::on_buttonZeroAll_pressed()
{
   
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice(m_dmName);
   ip.setName("zeroAll");
   ip.add(pcf::IndiElement("request"));
   
   ip["request"].setSwitchState(pcf::IndiElement::On);
   
   sendNewProperty(ip);   
}

void dmCtrl::on_buttonZero_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("zeroDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonRelease_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("releaseDM");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);
}



void dmCtrl::on_comboSelectFlat_activated(int index)
{
   std::string choice = ui.comboSelectFlat->itemText(index).toStdString();
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("flat");
   
   for(int i=0; i < ui.comboSelectFlat->count(); ++i)
   {
      std::string eln = ui.comboSelectFlat->itemText(i).toStdString();
      std::cerr << eln << "\n";
      ipFreq.add(pcf::IndiElement(eln));
      if(eln == choice) ipFreq[eln] = pcf::IndiElement::On;
      else ipFreq[eln] = pcf::IndiElement::Off;
   }
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonSetFlat_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("flat_set");
   ipFreq.add(pcf::IndiElement("toggle"));
   ipFreq["toggle"] = pcf::IndiElement::On;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonZeroFlat_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("flat_set");
   ipFreq.add(pcf::IndiElement("toggle"));
   ipFreq["toggle"] = pcf::IndiElement::Off;
    
   sendNewProperty(ipFreq);
}



void dmCtrl::on_comboSelectTest_activated(int index)
{
   std::string choice = ui.comboSelectTest->itemText(index).toStdString();
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("test");
   
   for(int i=0; i < ui.comboSelectTest->count(); ++i)
   {
      std::string eln = ui.comboSelectTest->itemText(i).toStdString();
      ipFreq.add(pcf::IndiElement(eln));
      if(eln == choice) ipFreq[eln] = pcf::IndiElement::On;
      else ipFreq[eln] = pcf::IndiElement::Off;
   }
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonSetTest_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("test_set");
   ipFreq.add(pcf::IndiElement("toggle"));
   ipFreq["toggle"] = pcf::IndiElement::On;
    
   sendNewProperty(ipFreq);
}

void dmCtrl::on_buttonZeroTest_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_dmName);
   ipFreq.setName("test_set");
   ipFreq.add(pcf::IndiElement("toggle"));
   ipFreq["toggle"] = pcf::IndiElement::Off;
    
   sendNewProperty(ipFreq);
}

} //namespace xqt
   
#include "moc_dmCtrl.cpp"

#endif
