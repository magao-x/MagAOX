#ifndef camera_hpp
#define camera_hpp


#include "ui_camera.h"

#include "xWidgets/xWidget.hpp"
#include "xWidgets/fsmDisplay.hpp"
#include "xWidgets/statusEntry.hpp"
#include "xWidgets/statusDisplay.hpp"
#include "xWidgets/selectionSwStatus.hpp"

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

   fsmDisplay * ui_fsmState {nullptr};

   statusEntry * ui_tempCCD {nullptr};

   statusDisplay * ui_tempStatus {nullptr};

   shutterStatus * ui_shutterStatus {nullptr};

   roiStatus * ui_roiStatus {nullptr};
   selectionSwStatus * ui_modes {nullptr};

   selectionSwStatus * ui_readoutSpd {nullptr};
   selectionSwStatus * ui_vshiftSpd {nullptr};

   statusEntry * ui_expTime {nullptr};
   statusEntry * ui_fps {nullptr};
   statusEntry * ui_emGain {nullptr};

   float m_temp {-99};

   bool m_inUpdate {false};

   QTimer * m_updateTimer {nullptr}; ///< Timer for periodic updates

public:
   camera( std::string & camName,
           QWidget * Parent = 0, 
           Qt::WindowFlags f = 0
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

   void setup_shutter();

   void setup_roiStatus();
   void setup_modes();

   void setup_readoutSpd();
   void setup_vshiftSpd();

   void setup_expTime(bool ro);
   void setup_fps(bool ro);
   void setup_emGain(bool ro);

signals:
   void updateTimerStop();
   void updateTimerStart(int);

   void add_temp_ccd(bool ro);
   void add_tempStatus();

   void add_shutter();

   void add_roiStatus();
   void add_modes();

   void add_readoutSpd();
   void add_vshiftSpd();

   void add_expTime(bool ro);   
   void add_fps(bool ro);
   void add_emGain(bool ro);

private:
     
   Ui::camera ui;
};
   
camera::camera( std::string & camName,
                QWidget * Parent, 
                Qt::WindowFlags f) : xWidget(Parent, f), m_camName{camName}
{
   ui.setupUi(this);

   m_updateTimer = new QTimer(this);

   connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   connect(this, SIGNAL(updateTimerStop()), m_updateTimer, SLOT(stop()));
   connect(this, SIGNAL(updateTimerStart(int)), m_updateTimer, SLOT(start(int)));

   connect(this, SIGNAL(add_temp_ccd(bool)), this, SLOT(setup_temp_ccd(bool)));
   connect(this, SIGNAL(add_tempStatus()), this, SLOT(setup_tempStatus()));

   connect(this, SIGNAL(add_shutter()), this, SLOT(setup_shutter()));

   connect(this, SIGNAL(add_roiStatus()), this, SLOT(setup_roiStatus()));
   connect(this, SIGNAL(add_modes()), this, SLOT(setup_modes()));

   connect(this, SIGNAL(add_readoutSpd()), this, SLOT(setup_readoutSpd()));
   connect(this, SIGNAL(add_vshiftSpd()), this, SLOT(setup_vshiftSpd()));

   connect(this, SIGNAL(add_expTime(bool)), this, SLOT(setup_expTime(bool)));
   connect(this, SIGNAL(add_fps(bool)), this, SLOT(setup_fps(bool)));
   connect(this, SIGNAL(add_emGain(bool)), this, SLOT(setup_emGain(bool)));
   
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

   m_parent->addSubscriber(ui_fsmState);

   if(ui_tempCCD) m_parent->addSubscriber(ui_tempCCD);
   if(ui_tempStatus) m_parent->addSubscriber(ui_tempStatus);
   if(ui_shutterStatus) m_parent->addSubscriber(ui_shutterStatus);
   if(ui_roiStatus) m_parent->addSubscriber(ui_roiStatus);
   if(ui_modes) m_parent->addSubscriber(ui_modes);
   if(ui_readoutSpd) m_parent->addSubscriber(ui_readoutSpd);
   if(ui_vshiftSpd) m_parent->addSubscriber(ui_vshiftSpd);
   if(ui_expTime) m_parent->addSubscriber(ui_expTime);
   if(ui_fps) m_parent->addSubscriber(ui_fps);
   if(ui_emGain) m_parent->addSubscriber(ui_emGain);

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

   if(ui_expTime) ui_expTime->onConnect();
   if(ui_fps) ui_fps->onConnect();
   if(ui_emGain) ui_emGain->onConnect();
   
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

   if(ui_expTime) ui_expTime->onDisconnect();
   if(ui_fps) ui_fps->onDisconnect();
   if(ui_emGain) ui_emGain->onDisconnect();

   clearFocus();   

   setEnableDisable(false);
}

void camera::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void camera::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_camName) return;
   
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
   if(ui_expTime) ui_expTime->updateGUI();
   if(ui_fps) ui_fps->updateGUI();
   if(ui_emGain) ui_emGain->updateGUI();

   emit updateTimerStart(1000);
   m_inUpdate = false;

} //updateGUI()

void camera::setup_temp_ccd(bool ro)
{
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
   ui_tempStatus = new statusDisplay(m_camName,"temp_control", "status", "Temp. Ctrl.", "", this, 0);
   ui_tempStatus->setObjectName(QString::fromUtf8("tempStatus"));
   
   ui.grid->addWidget(ui_tempStatus, 1, 1, 1, 1);
   
   ui_tempStatus->onDisconnect();

   m_parent->addSubscriber(ui_tempStatus);
}

void camera::setup_shutter()
{
   ui_shutterStatus = new shutterStatus(m_camName, this);
   ui_shutterStatus->setObjectName(QString::fromUtf8("shutter"));
   
   ui.grid->addWidget(ui_shutterStatus, 6, 0, 2, 1);
   
   ui_shutterStatus->onDisconnect();

   m_parent->addSubscriber(ui_shutterStatus);
}

void camera::setup_roiStatus()
{
   ui_roiStatus = new roiStatus(m_camName, this);
   ui_roiStatus->setObjectName(QString::fromUtf8("roiStatus"));
   
   ui.grid->addWidget(ui_roiStatus, 3, 1, 1, 1);
   
   ui_roiStatus->onDisconnect();

   m_parent->addSubscriber(ui_roiStatus);
}

void camera::setup_modes()
{
   ui_modes = new selectionSwStatus(m_camName,"mode", "", "Mode", "", this);
   ui_modes->setObjectName(QString::fromUtf8("modes"));
   
   ui.grid->addWidget(ui_modes, 3, 1, 1, 1);
   
   ui_modes->onDisconnect();

   m_parent->addSubscriber(ui_modes);
}

void camera::setup_readoutSpd()
{
   ui_readoutSpd = new selectionSwStatus(m_camName,"readout_speed", "", "Readout Spd", "", this);
   ui_readoutSpd->setObjectName(QString::fromUtf8("readoutSpd"));
   
   ui.grid->addWidget(ui_readoutSpd, 4, 1, 1, 1);
   
   ui_readoutSpd->onDisconnect();

   m_parent->addSubscriber(ui_readoutSpd);
}

void camera::setup_vshiftSpd()
{
   ui_vshiftSpd = new selectionSwStatus(m_camName,"vshift_speed", "", "Vert. Shift Spd", "", this);
   ui_vshiftSpd->setObjectName(QString::fromUtf8("vshiftSpd"));
   
   ui.grid->addWidget(ui_vshiftSpd, 5, 1, 1, 1);
   
   ui_vshiftSpd->onDisconnect();

   m_parent->addSubscriber(ui_vshiftSpd);
}

void camera::setup_expTime(bool ro)
{
   ui_expTime = new statusEntry(this);
   ui_expTime->setObjectName(QString::fromUtf8("expTime"));
   ui_expTime->setup(m_camName, "exptime", statusEntry::FLOAT, "Exp. Time", "sec");
   ui_expTime->highlightChanges(true);
   ui_expTime->readOnly(ro);

   ui.grid->addWidget(ui_expTime, 6, 1, 1, 1);
   
   ui_expTime->onDisconnect();

   m_parent->addSubscriber(ui_expTime);
}

void camera::setup_fps(bool ro)
{
   ui_fps = new statusEntry(this);
   ui_fps->setObjectName(QString::fromUtf8("fps"));
   ui_fps->setup(m_camName, "fps", statusEntry::FLOAT, "Frame Rate", "F.P.S.");
   ui_fps->highlightChanges(true);
   ui_fps->readOnly(ro);

   ui.grid->addWidget(ui_fps, 7, 1, 1, 1);
   
   ui_fps->onDisconnect();

   m_parent->addSubscriber(ui_fps);
}

void camera::setup_emGain(bool ro)
{
   ui_emGain = new statusEntry(this);
   ui_emGain->setObjectName(QString::fromUtf8("emgain"));
   ui_emGain->setup(m_camName, "emgain", statusEntry::FLOAT, "E.M. Gain", "");
   ui_emGain->highlightChanges(true);
   ui_emGain->readOnly(ro);

   ui.grid->addWidget(ui_emGain, 8, 1, 1, 1);
   
   ui_emGain->onDisconnect();

   m_parent->addSubscriber(ui_emGain);
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

   
   if(ui_tempCCD) ui_tempCCD->setEnabled(tf);
   if(ui_tempStatus) ui_tempStatus->setEnabled(tf);

   if(ui_roiStatus) ui_roiStatus->setEnabled(tf);
   if(ui_modes) ui_modes->setEnabled(tf);
   if(ui_readoutSpd) ui_readoutSpd->setEnabled(tf);
   if(ui_vshiftSpd) ui_vshiftSpd->setEnabled(tf);
   if(ui_expTime) ui_expTime->setEnabled(tf);
   if(ui_fps) ui_fps->setEnabled(tf);
   if(ui_emGain) ui_emGain->setEnabled(tf);
   
}





} //namespace xqt
   
#include "moc_camera.cpp"

#endif
