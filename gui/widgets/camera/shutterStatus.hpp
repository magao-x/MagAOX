#ifndef shutterStatus_hpp
#define shutterStatus_hpp

#include "ui_shutterStatus.h"

#include "../xWidgets/xWidget.hpp"
#include "../../lib/multiIndiSubscriber.hpp"

namespace xqt 
{
   
/// Widget to display a camera's shutter status and allow changing
class shutterStatus : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_camName; ///< INDI device name of camera

   std::string m_status; ///< Current status of the shutter (OFF, READY, etc.)
   int m_state {0}; ///< Current state of the shutter, 0 is open, 1 is shut.
   int m_tgt_state {-1}; ///< Target state of the shutter in this widget.  Used to avoid bouncing gui.
   

public:
   explicit shutterStatus( const std::string & camName,
                           QWidget * Parent = 0, 
                           Qt::WindowFlags f = Qt::WindowFlags()
                         );
   
   ~shutterStatus();

   virtual void subscribe();
             
   virtual void onConnect();

   virtual void onDisconnect();
   
   virtual void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   virtual void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   virtual void updateGUI();


protected:
     
   Ui::shutterStatus ui;
};
   
shutterStatus::shutterStatus( const std::string & camName,
                              QWidget * Parent, 
                              Qt::WindowFlags f) : xWidget(Parent, f), m_camName{camName}
{
   ui.setupUi(this);

   ui.shutter->setup(camName, "shutter", "toggle", "");
   ui.shutter->setStretch(0,0,3, true, true);

   onDisconnect();
}
   
shutterStatus::~shutterStatus()
{
}

void shutterStatus::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriber(ui.shutter);
   m_parent->addSubscriberProperty(this, m_camName, "shutter_status");

   return;
}
  
void shutterStatus::onConnect()
{
}

void shutterStatus::onDisconnect()
{
   ui.label->setText("shutter (disconnected)");
}

void shutterStatus::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void shutterStatus::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_camName) return;
   
   if(ipRecv.getName() == "shutter")
   {
      if(ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"] == pcf::IndiElement::On) m_state = 1;
         else m_state = 0;

         if(m_tgt_state == -1) m_tgt_state = m_state;
      }
   }

   if(ipRecv.getName() == "shutter_status")
   {
      if(ipRecv.find("status"))
      {
         m_status = ipRecv["status"].get();
      }
   }

   updateGUI();
}

void shutterStatus::updateGUI()
{
   if(m_status == "READY")
   {
      ui.label->setEnabled(true);
      ui.shutter->setEnabled(true);

      ui.label->setText("shutter");
   }
   else if (m_status == "POWEROFF")
   {
      ui.label->setEnabled(true);
      ui.shutter->setEnabled(false);

      ui.label->setText("shutter (off)");
   }
   else
   {
      ui.label->setEnabled(true);
      ui.shutter->setEnabled(false);

      ui.label->setText("shutter (?)");
   }

} //updateGUI()



} //namespace xqt
   
#include "moc_shutterStatus.cpp"

#endif
