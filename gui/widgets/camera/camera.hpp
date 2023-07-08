#ifndef camera_hpp
#define camera_hpp


#include "ui_camera.h"

#include "xWidgets/xWidget.hpp"
#include "xWidgets/fsmDisplay.hpp"
#include "xWidgets/statusEntry.hpp"
#include "xWidgets/statusDisplay.hpp"
#include "xWidgets/selectionSwStatus.hpp"
#include "xWidgets/toggleSlider.hpp"

#include "roi/roi.hpp"

#include "camera/roiStatus.hpp"
#include "camera/shutterStatus.hpp"

namespace xqt 
{
   
class camera : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   std::string m_camName;
   std::string m_darkName;

   fsmDisplay * ui_fsmState {nullptr};

   statusEntry * ui_tempCCD {nullptr};

   statusDisplay * ui_tempStatus {nullptr};

   QPushButton * ui_reconfigure {nullptr};

   shutterStatus * ui_shutterStatus {nullptr};
   //toggleSlider * ui_shutterStatus {nullptr};

   roiStatus * ui_roiStatus {nullptr};
   selectionSwStatus * ui_modes {nullptr};

   selectionSwStatus * ui_readoutSpd {nullptr};
   selectionSwStatus * ui_vshiftSpd {nullptr};

   toggleSlider * ui_cropMode {nullptr};

   statusEntry * ui_expTime {nullptr};
   statusEntry * ui_fps {nullptr};
   statusEntry * ui_emGain {nullptr};

   toggleSlider * ui_synchro {nullptr};

   QPushButton * ui_takeDarks {nullptr};


   float m_temp {-99};

   bool m_takingDark {false};

   bool m_inUpdate {false};

   QTimer * m_updateTimer {nullptr}; ///< Timer for periodic updates

public:
   explicit camera( std::string & camName,
                    QWidget * Parent = 0, 
                    Qt::WindowFlags f = Qt::WindowFlags()
                  );
   
   ~camera();
   
   void subscribe( );
             
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void hideAll();
   
   void setEnableDisable(bool tf, bool all=true);

   //void clearFocus();

public slots:
   void updateGUI();
   
   void setup_temp_ccd(bool ro);
   void setup_tempStatus();

   void setup_reconfigure();
   void reconfigure();
   void setup_shutter();

   void setup_roiStatus();
   void setup_modes();

   void setup_readoutSpd();
   void setup_vshiftSpd();

   void setup_cropMode();

   void setup_expTime(bool ro);
   void setup_fps(bool ro);
   void setup_emGain(bool ro);

   void setup_synchro();

   void setup_takeDarks();

   void takeDark();

signals:
   void updateTimerStop();
   void updateTimerStart(int);

   void add_temp_ccd(bool ro);
   void add_tempStatus();

   void add_reconfigure();

   void add_shutter();

   void add_roiStatus();
   void add_modes();

   void add_readoutSpd();
   void add_vshiftSpd();

   void add_cropMode();

   void add_expTime(bool ro);   
   void add_fps(bool ro);
   void add_emGain(bool ro);

   void add_synchro();

   void add_takeDarks();
private:
     
   Ui::camera ui;
};
   
camera::camera( std::string & camName,
                QWidget * Parent, 
                Qt::WindowFlags f) : xWidget(Parent, f), m_camName{camName}
{
   m_darkName = m_camName + "-dark";
   ui.setupUi(this);

   m_updateTimer = new QTimer(this);

   connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   connect(this, SIGNAL(updateTimerStop()), m_updateTimer, SLOT(stop()));
   connect(this, SIGNAL(updateTimerStart(int)), m_updateTimer, SLOT(start(int)));

   connect(this, SIGNAL(add_temp_ccd(bool)), this, SLOT(setup_temp_ccd(bool)));
   connect(this, SIGNAL(add_tempStatus()), this, SLOT(setup_tempStatus()));

   connect(this, SIGNAL(add_reconfigure()), this, SLOT(setup_reconfigure()));

   connect(this, SIGNAL(add_shutter()), this, SLOT(setup_shutter()));

   connect(this, SIGNAL(add_roiStatus()), this, SLOT(setup_roiStatus()));
   connect(this, SIGNAL(add_modes()), this, SLOT(setup_modes()));

   connect(this, SIGNAL(add_readoutSpd()), this, SLOT(setup_readoutSpd()));
   connect(this, SIGNAL(add_vshiftSpd()), this, SLOT(setup_vshiftSpd()));
   connect(this, SIGNAL(add_cropMode()), this, SLOT(setup_cropMode()));

   connect(this, SIGNAL(add_expTime(bool)), this, SLOT(setup_expTime(bool)));
   connect(this, SIGNAL(add_fps(bool)), this, SLOT(setup_fps(bool)));
   connect(this, SIGNAL(add_emGain(bool)), this, SLOT(setup_emGain(bool)));
   connect(this, SIGNAL(add_synchro()), this, SLOT(setup_synchro()));

   connect(this, SIGNAL(add_takeDarks()), this, SLOT(setup_takeDarks()));

   QSpacerItem *holder = new QSpacerItem(10,0, QSizePolicy::Expanding, QSizePolicy::Expanding);
   ui.grid->addItem(holder, 2,1,1,1);

   ui_fsmState = new xqt::fsmDisplay(this);
   ui_fsmState->setObjectName(QString::fromUtf8("fsmState"));
   ui.grid->addWidget(ui_fsmState, 1, 0, 1, 1);
   ui_fsmState->device(m_camName);

   QFont qf = ui.lab_camName->font();
   qf.setPixelSize(XW_FONT_SIZE+3);
   ui.lab_camName->setFont(qf);

   ui.lab_camName->setText(m_camName.c_str());

   
   onDisconnect();
}
   
camera::~camera()
{
}

void camera::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_camName, "fsm");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_camName, "reconfigure");
   m_parent->addSubscriberProperty((multiIndiSubscriber *) this, m_darkName, "start");

   m_parent->addSubscriber(ui_fsmState);

   if(ui_tempCCD) m_parent->addSubscriber(ui_tempCCD);
   if(ui_tempStatus) m_parent->addSubscriber(ui_tempStatus);
   if(ui_shutterStatus) m_parent->addSubscriber(ui_shutterStatus);
   if(ui_roiStatus) m_parent->addSubscriber(ui_roiStatus);
   if(ui_modes) m_parent->addSubscriber(ui_modes);
   if(ui_readoutSpd) m_parent->addSubscriber(ui_readoutSpd);
   if(ui_vshiftSpd) m_parent->addSubscriber(ui_vshiftSpd);
   if(ui_cropMode) m_parent->addSubscriber(ui_cropMode);
   if(ui_expTime) m_parent->addSubscriber(ui_expTime);
   if(ui_fps) m_parent->addSubscriber(ui_fps);
   if(ui_emGain) m_parent->addSubscriber(ui_emGain);
   if(ui_synchro) m_parent->addSubscriber(ui_synchro);

   return;
}
  
void camera::onConnect()
{
   ui.lab_camName->setEnabled(true);

   setWindowTitle(QString((m_camName+"Ctrl").c_str()));

   ui_fsmState->onConnect();

   if(ui_tempCCD) ui_tempCCD->onConnect();
   if(ui_tempStatus) ui_tempStatus->onConnect();
   if(ui_shutterStatus) ui_shutterStatus->onConnect();

   if(ui_roiStatus) ui_roiStatus->onConnect();
   if(ui_modes) ui_modes->onConnect();
   if(ui_readoutSpd) ui_readoutSpd->onConnect();
   if(ui_vshiftSpd) ui_readoutSpd->onConnect();
   if(ui_cropMode) ui_cropMode->onConnect();

   if(ui_expTime) ui_expTime->onConnect();
   if(ui_fps) ui_fps->onConnect();
   if(ui_emGain) ui_emGain->onConnect();
   
   if(ui_synchro) ui_synchro->onConnect();

   clearFocus();

   //updateGUI();
}

void camera::onDisconnect()
{

   setWindowTitle(QString((m_camName+"Ctrl").c_str()) + QString(" (disconnected)"));

   ui_fsmState->onDisconnect();

   if(ui_tempCCD) ui_tempCCD->onDisconnect();
   if(ui_tempStatus) ui_tempStatus->onDisconnect();
   if(ui_shutterStatus) ui_shutterStatus->onDisconnect();

   if(ui_roiStatus) ui_roiStatus->onDisconnect();
   if(ui_modes) ui_modes->onDisconnect();
   if(ui_readoutSpd) ui_readoutSpd->onDisconnect();
   if(ui_vshiftSpd) ui_readoutSpd->onDisconnect();
   if(ui_cropMode) ui_cropMode->onDisconnect();

   if(ui_expTime) ui_expTime->onDisconnect();
   if(ui_fps) ui_fps->onDisconnect();
   if(ui_emGain) ui_emGain->onDisconnect();

   if(ui_synchro) ui_synchro->onDisconnect();

   clearFocus();   

   setEnableDisable(false);
}

void camera::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void camera::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_camName && ipRecv.getDevice() != m_darkName) return;
   
   if(ipRecv.getDevice() == m_camName)
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_appState = ipRecv["state"].get<std::string>();
         }
      }
      
      if(ipRecv.getName() == "temp_ccd")
      {
         if(!ui_tempCCD)
         {
            bool ro = true;
            if(ipRecv.find("target")) ro = false;
   
            emit add_temp_ccd(ro);
         }
      }
   
      if(ipRecv.getName() == "temp_control")
      {
         if(!ui_tempStatus)
         {
            emit add_tempStatus();
         }
      }
   
      if(ipRecv.getName() == "reconfigure")
      {
         if(!ui_reconfigure)
         {
            emit add_reconfigure();
         }
      }

      if(ipRecv.getName() == "shutter")
      {
         if(!ui_shutterStatus)
         {
            emit add_shutter();
         }
      }
   
      if(ipRecv.getName() == "roi_set")
      {
         if(!ui_roiStatus)
         {
            emit add_roiStatus();
         }
      }
   
      if(ipRecv.getName() == "mode")
      {
         if(!ui_modes)
         {
            emit add_modes();
         }
      }
   
      if(ipRecv.getName() == "readout_speed")
      {
         if(!ui_readoutSpd)
         {
            emit add_readoutSpd();
         }
      }
   
      if(ipRecv.getName() == "vshift_speed")
      {
         if(!ui_vshiftSpd)
         {
            emit add_vshiftSpd();
         }
      }
   
      if(ipRecv.getName() == "roi_crop_mode")
      {
         if(!ui_cropMode)
         {
            emit add_cropMode();
         }
      }

      if(ipRecv.getName() == "exptime")
      {
         if(!ui_expTime)
         {
            bool ro = true;
            if(ipRecv.find("target")) ro = false;
   
            emit add_expTime(ro);
         }
      }
   
      if(ipRecv.getName() == "fps")
      {
         if(!ui_fps)
         {
            bool ro = true;
            if(ipRecv.find("target")) ro = false;
   
            emit add_fps(ro);
         }
      }
   
      if(ipRecv.getName() == "emgain")
      {
         if(!ui_emGain)
         {
            bool ro = true;
            if(ipRecv.find("target")) ro = false;
   
            emit add_emGain(ro);
         }
      }

      if(ipRecv.getName() == "synchro")
      {
         if(!ui_synchro)
         {   
            emit add_synchro();
         }
      }
      
   }
   else if(ipRecv.getDevice() == m_darkName)
   {
      if(!ui_takeDarks) emit add_takeDarks();

      if(ipRecv.getName() == "start" && ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
         {
            m_takingDark = true;
         }
         else 
         {
            m_takingDark = false;
         }
      }
   }

   updateGUI();   
}

void camera::updateGUI()
{
   if(m_inUpdate) return;
   emit updateTimerStop();
   m_inUpdate = true;

   if( m_appState == "NODEVICE" || m_appState == "NOTCONNECTED" || m_appState == "CONNECTED")
   {
      setEnableDisable(false, false);
      ui.lab_camName->setEnabled(true);
      ui_fsmState->setEnabled(true);
   }
   else if( m_appState != "READY" && m_appState != "OPERATING" && m_appState != "CONFIGURING")
   {
      setEnableDisable(false); 
      m_inUpdate = false;

      emit updateTimerStart(1000);
      return;     
   }
   else //if( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "CONFIGURING")
   {
      setEnableDisable(true);
   }
   
   //Update the component GUIs to ensure they update for connection state, etc.
   if(ui_tempCCD) ui_tempCCD->updateGUI();
   if(ui_roiStatus) ui_roiStatus->updateGUI();
   if(ui_shutterStatus) ui_shutterStatus->updateGUI();
   if(ui_modes) ui_modes->updateGUI();
   if(ui_readoutSpd) ui_readoutSpd->updateGUI();
   if(ui_vshiftSpd) ui_vshiftSpd->updateGUI();
   if(ui_cropMode) ui_cropMode->updateGUI();
   if(ui_expTime) ui_expTime->updateGUI();
   if(ui_fps) ui_fps->updateGUI();
   if(ui_emGain) ui_emGain->updateGUI();
   if(ui_synchro) ui_synchro->updateGUI();

   if( (m_appState == "READY" || m_appState == "OPERATING") && ui_takeDarks )
   {
      if(m_takingDark)
      {
         ui_takeDarks->setEnabled(false);
      }
      else
      {
         ui_takeDarks->setEnabled(true);
      }
   }

   emit updateTimerStart(1000);
   m_inUpdate = false;

} //updateGUI()

void camera::setup_temp_ccd(bool ro)
{
   if(ui_tempCCD) return;
   
   ui_tempCCD = new statusEntry(this);
   ui_tempCCD->setObjectName(QString::fromUtf8("tempCCD"));
   ui_tempCCD->setup(m_camName, "temp_ccd", statusEntry::FLOAT, "Detector Temp.", "C");
   ui_tempCCD->highlightChanges(false);
   ui_tempCCD->readOnly(ro);

   ui.grid->addWidget(ui_tempCCD, 0, 1, 1, 1);
   
   ui_tempCCD->onDisconnect();

   m_parent->addSubscriber(ui_tempCCD);
}

void camera::setup_tempStatus()
{
   if(ui_tempStatus) return;
   
   ui_tempStatus = new statusDisplay(m_camName,"temp_control", "status", "Temp. Ctrl.", "", this, Qt::WindowFlags());
   ui_tempStatus->setObjectName(QString::fromUtf8("tempStatus"));
   
   ui.grid->addWidget(ui_tempStatus, 1, 1, 1, 1);
   
   ui_tempStatus->onDisconnect();

   m_parent->addSubscriber(ui_tempStatus);
}

void camera::setup_reconfigure()
{
   if(ui_reconfigure) return;
   
   ui_reconfigure = new QPushButton(this);
   ui_reconfigure->setObjectName(QString::fromUtf8("reconfigure"));
   ui_reconfigure->setText("reconfigure");
   ui_reconfigure->setMaximumWidth(200);
   connect(ui_reconfigure, SIGNAL(pressed()), this, SLOT(reconfigure()));
   ui.grid->addWidget(ui_reconfigure, 3, 0, 1, 1, Qt::AlignHCenter);   

}

void camera::reconfigure()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_camName);
   ipFreq.setName("reconfigure");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void camera::setup_shutter()
{
   if(ui_shutterStatus) return;
   
   ui_shutterStatus = new shutterStatus(m_camName, this);

   //ui_shutterStatus = new toggleSlider(m_camName, "shutter", "toggle", "Shutter");
   ui_shutterStatus->setObjectName(QString::fromUtf8("shutter"));
   
   ui.grid->addWidget(ui_shutterStatus, 7, 0, 2, 1);
   
   ui_shutterStatus->onDisconnect();

   m_parent->addSubscriber(ui_shutterStatus);
}

void camera::setup_roiStatus()
{
   if(ui_roiStatus) return; //can get called from several threads
   
   ui_roiStatus = new roiStatus(m_camName, this);
   ui_roiStatus->setObjectName(QString::fromUtf8("roiStatus"));
   
   ui.grid->addWidget(ui_roiStatus, 3, 1, 1, 1);
   
   ui_roiStatus->onDisconnect();

   m_parent->addSubscriber(ui_roiStatus);
}

void camera::setup_modes()
{
   if(ui_modes) return;
   
   ui_modes = new selectionSwStatus(m_camName,"mode", "", "Mode", "", this);
   ui_modes->setObjectName(QString::fromUtf8("modes"));
   
   ui.grid->addWidget(ui_modes, 3, 1, 1, 1);
   
   ui_modes->onDisconnect();

   m_parent->addSubscriber(ui_modes);
}

void camera::setup_readoutSpd()
{
   if(ui_readoutSpd) return;
   
   ui_readoutSpd = new selectionSwStatus(m_camName,"readout_speed", "", "Readout Spd", "", this);
   ui_readoutSpd->setObjectName(QString::fromUtf8("readoutSpd"));
   
   ui.grid->addWidget(ui_readoutSpd, 4, 1, 1, 1);
   
   ui_readoutSpd->onDisconnect();

   m_parent->addSubscriber(ui_readoutSpd);
}

void camera::setup_vshiftSpd()
{
   if(ui_vshiftSpd) return;
   
   ui_vshiftSpd = new selectionSwStatus(m_camName,"vshift_speed", "", "Vert. Shift Spd", "", this);
   ui_vshiftSpd->setObjectName(QString::fromUtf8("vshiftSpd"));
   
   ui.grid->addWidget(ui_vshiftSpd, 5, 1, 1, 1);
   
   ui_vshiftSpd->onDisconnect();

   m_parent->addSubscriber(ui_vshiftSpd);
}

void camera::setup_cropMode()
{
   if(ui_cropMode) return;
   
   ui_cropMode = new toggleSlider(m_camName, "roi_crop_mode", "Crop Mode", this);
   ui_cropMode->setObjectName(QString::fromUtf8("cropMode"));

   ui.grid->addWidget(ui_cropMode, 6, 1, 1, 1);
   
   ui_cropMode->onDisconnect();

   m_parent->addSubscriber(ui_cropMode);
}

void camera::setup_expTime(bool ro)
{
   if(ui_expTime) return;
   
   ui_expTime = new statusEntry(this);
   ui_expTime->setObjectName(QString::fromUtf8("expTime"));
   ui_expTime->setup(m_camName, "exptime", statusEntry::FLOAT, "Exp. Time", "sec");
   ui_expTime->highlightChanges(true);
   ui_expTime->readOnly(ro);

   ui.grid->addWidget(ui_expTime, 7, 1, 1, 1);
   
   ui_expTime->onDisconnect();

   m_parent->addSubscriber(ui_expTime);
}

void camera::setup_fps(bool ro)
{
   if(ui_fps) return;
   
   ui_fps = new statusEntry(this);
   ui_fps->setObjectName(QString::fromUtf8("fps"));
   ui_fps->setup(m_camName, "fps", statusEntry::FLOAT, "Frame Rate", "F.P.S.");
   ui_fps->highlightChanges(true);
   ui_fps->readOnly(ro);

   ui.grid->addWidget(ui_fps, 8, 1, 1, 1);
   
   ui_fps->onDisconnect();

   m_parent->addSubscriber(ui_fps);
}

void camera::setup_emGain(bool ro)
{
   if(ui_emGain) return;
   
   ui_emGain = new statusEntry(this);
   ui_emGain->setObjectName(QString::fromUtf8("emgain"));
   ui_emGain->setup(m_camName, "emgain", statusEntry::FLOAT, "E.M. Gain", "");
   ui_emGain->highlightChanges(true);
   ui_emGain->readOnly(ro);

   ui.grid->addWidget(ui_emGain, 9, 1, 1, 1);
   
   ui_emGain->onDisconnect();

   m_parent->addSubscriber(ui_emGain);
}

void camera::setup_synchro()
{
   if(ui_synchro) return;
   
   ui_synchro = new toggleSlider(m_camName, "synchro", "Synchro", this);
   ui_synchro->setObjectName(QString::fromUtf8("synchro"));

   ui.grid->addWidget(ui_synchro, 10, 1, 1, 1);
   
   ui_synchro->onDisconnect();

   m_parent->addSubscriber(ui_synchro);
}

void camera::setup_takeDarks()
{
   if(ui_takeDarks) return;
   
   ui_takeDarks = new QPushButton(this);
   ui_takeDarks->setObjectName(QString::fromUtf8("takeDarks"));
   ui_takeDarks->setText("take darks");
   ui_takeDarks->setMaximumWidth(200);
   ui_takeDarks->setFocusPolicy(Qt::NoFocus);
   connect(ui_takeDarks, SIGNAL(pressed()), this, SLOT(takeDark()));
   ui.grid->addWidget(ui_takeDarks, 9, 0, 1, 1,Qt::AlignHCenter);   

}

void camera::takeDark()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_darkName);
   ipFreq.setName("start");
   ipFreq.add(pcf::IndiElement("toggle"));
   ipFreq["toggle"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void camera::hideAll()
{
   if(ui_roiStatus) ui_roiStatus->hide();
}

void camera::setEnableDisable(bool tf, bool all)
{
   if(all)
   {
      ui.lab_camName->setEnabled(tf);
      ui_fsmState->setEnabled(tf);
   }

   if(ui_reconfigure) ui_reconfigure->setEnabled(tf);
   if(ui_tempCCD) ui_tempCCD->setEnabled(tf);
   if(ui_tempStatus) ui_tempStatus->setEnabled(tf);

   if(ui_roiStatus) ui_roiStatus->setEnabled(tf);
   if(ui_modes) ui_modes->setEnabled(tf);
   if(ui_readoutSpd) ui_readoutSpd->setEnabled(tf);
   if(ui_vshiftSpd) ui_vshiftSpd->setEnabled(tf);
   if(ui_expTime) ui_expTime->setEnabled(tf);
   if(ui_fps) ui_fps->setEnabled(tf);
   if(ui_emGain) ui_emGain->setEnabled(tf);
   if(ui_shutterStatus) ui_shutterStatus->setEnabled(tf);
   
   if(ui_takeDarks) ui_takeDarks->setEnabled(tf);
   
}

} //namespace xqt
   
#include "moc_camera.cpp"

#endif
