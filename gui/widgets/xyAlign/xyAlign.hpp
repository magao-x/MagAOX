#ifndef xyAlign_hpp
#define xyAlign_hpp

#include <cmath>
#include <unistd.h>

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_xyAlign.h"

#include "../xWidgets/xWidget.hpp"


namespace xqt 
{
   
class xyAlign : public xWidget
{
   Q_OBJECT
   
protected:
   QMutex m_mutex;
   
   std::string m_title;

   //Device state
   std::string m_device{"gmtpicos"};
   std::string m_xProperty{"pico2_pos"};
   std::string m_yProperty{"pico1_pos"};
   std::string m_fsmState;
   
   double m_xPos;
   double m_yPos;

   double m_stepSize {100};   
   int m_scale {100};

   
public:
   xyAlign( QWidget * Parent = 0, 
               Qt::WindowFlags f = Qt::WindowFlags()
             );
   
   ~xyAlign();
   
   void subscribe();
                               
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void enableButtons();
   void disableButtons();
   
public slots:
   void updateGUI();
   
   void on_button_u_pressed();
   void on_button_d_pressed();
   void on_button_l_pressed();
   void on_button_r_pressed();
   void on_button_scale_pressed();
      
private:
     
   Ui::xyAlign ui;
};

xyAlign::xyAlign( QWidget * Parent, Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   ui.button_scale->setProperty("isScaleButton", true);
   
   QTimer *timer = new QTimer(this);
   connect(timer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   timer->start(250);
      
   char ss[5];
   snprintf(ss, 5, "%d", m_scale);
   ui.button_scale->setText(ss);
   
   ui.fsmState->device(m_device);

   ui.xPos->setup(m_device, m_xProperty, statusEntry::INT, "x", "");
   ui.xPos->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.xPos->format("%d");
   
   ui.yPos->setup(m_device, m_yProperty, statusEntry::INT, "y", "");
   ui.yPos->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.yPos->format("%d");

   setXwFont(ui.xPos);
   setXwFont(ui.yPos);

   onDisconnect();
}
   
xyAlign::~xyAlign()
{
}

void xyAlign::subscribe()
{
   if(m_parent == nullptr) return;

   m_parent->addSubscriberProperty(this, m_device, "fsm");
   
   m_parent->addSubscriberProperty(this, m_device, m_xProperty);
   m_parent->addSubscriberProperty(this, m_device, m_yProperty);
   
   m_parent->addSubscriber(ui.fsmState);
   m_parent->addSubscriber(ui.xPos);
   m_parent->addSubscriber(ui.yPos);

   return;
}
 
void xyAlign::onConnect()
{
   ui.fsmState->setEnabled(true);
   
   ui.button_u->setEnabled(true);
   ui.button_d->setEnabled(true);
   ui.button_l->setEnabled(true);
   ui.button_r->setEnabled(true);
   ui.button_scale->setEnabled(true);
      
   ui.fsmState->onConnect();
   ui.xPos->onConnect();
   ui.yPos->onConnect();

   setWindowTitle(m_title.c_str());
}

void xyAlign::onDisconnect()
{
   m_fsmState = "";
   
   ui.fsmState->setEnabled(false);
   
   ui.xPos->setEnabled(false);
   ui.yPos->setEnabled(false);

   ui.button_u->setEnabled(false);
   ui.button_d->setEnabled(false);
   ui.button_l->setEnabled(false);
   ui.button_r->setEnabled(false);
   ui.button_scale->setEnabled(false);
   
   ui.fsmState->onDisconnect();
   ui.xPos->onDisconnect();
   ui.yPos->onDisconnect();

   setWindowTitle(QString(m_title.c_str()) + " (disconnected)");
}
   
void xyAlign::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   std::string dev = ipRecv.getDevice();
   if( dev == m_device ) 
   {
      return handleSetProperty(ipRecv);
   }
   
   return;
}

void xyAlign::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::string dev = ipRecv.getDevice();
   
   if(dev == m_device)
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_fsmState  = ipRecv["state"].get<std::string>();
            return;
         }
      }
      if(ipRecv.getName() == m_xProperty)
      {
         if(ipRecv.find("current"))
         {
            m_xPos = ipRecv["current"].get<double>();
            return;
         }
      }
      if(ipRecv.getName() == m_yProperty)
      {
         if(ipRecv.find("current"))
         {
            m_xPos = ipRecv["current"].get<double>();
            return;
         }
      }
   }

   return;

}

void xyAlign::updateGUI()
{
   
   if(m_fsmState != "READY")
   {
      ui.xPos->setEnabled(false);

      disableButtons();
   }
   else
   {
      enableButtons();
   }
   
} //updateGUI()

void xyAlign::enableButtons()
{   
   ui.xPos->setEnabled(true);
   ui.yPos->setEnabled(true);

   ui.button_u->setEnabled(true);
   ui.button_d->setEnabled(true);
   ui.button_l->setEnabled(true);
   ui.button_r->setEnabled(true);
   ui.button_scale->setEnabled(true);

}

void xyAlign::disableButtons()
{
   ui.xPos->setEnabled(true);
   ui.yPos->setEnabled(true);

   ui.button_u->setEnabled(false);
   ui.button_d->setEnabled(false);
   ui.button_l->setEnabled(false);
   ui.button_r->setEnabled(false);
   ui.button_scale->setEnabled(false);

}

void xyAlign::on_button_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_device);
   ip.setName(m_yProperty);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_yPos + m_scale*m_stepSize;

   disableButtons();

   sendNewProperty(ip);
}

void xyAlign::on_button_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_device);
   ip.setName(m_yProperty);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_yPos - m_scale*m_stepSize;

   disableButtons();

   sendNewProperty(ip);
}

void xyAlign::on_button_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_device);
   ip.setName(m_xProperty);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_xPos - m_scale*m_stepSize;

   disableButtons();

   sendNewProperty(ip);
}

void xyAlign::on_button_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_device);
   ip.setName(m_xProperty);
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_xPos + m_scale*m_stepSize;

   disableButtons();

   sendNewProperty(ip);
   
}

void xyAlign::on_button_scale_pressed()
{
   if( m_scale == 100)
   {
      m_scale = 10;
   }
   else if(m_scale == 10)
   {
      m_scale = 5;
   }
   else if(m_scale == 5)
   {
      m_scale = 1;
   }
   else if(m_scale == 1)
   {
      m_scale = 100;
   }
   else
   {
      m_scale = 1;
   }
   
   char ss[5];
   snprintf(ss, 5, "%d", m_scale);
   std::cerr << m_scale << " " << ss << "\n";
   ui.button_scale->setText(ss);


}


} //namespace xqt
   
#include "moc_xyAlign.cpp"

#endif
