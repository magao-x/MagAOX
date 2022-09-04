#ifndef coronAlign_hpp
#define coronAlign_hpp

#include <cmath>
#include <unistd.h>

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_coronAlign.h"

#include "../xWidgets/xWidget.hpp"


namespace xqt 
{
   
class coronAlign : public xWidget
{
   Q_OBJECT
   
protected:
   QMutex m_mutex;
   
   //Pico Motors
   std::string m_picoState;
   
   //Pupil Plane
   
   std::string m_fwPupilState;
   
   
   double m_fwPupilPos;
   long m_picoPupilPos;
   
   double m_fwPupilStepSize {0.0001};
   double m_picoPupilStepSize {100};
   
   int m_pupilScale {1};

   //Focal Plane
   
   std::string m_fwFocalState;
   
   
   double m_fwFocalPos;
   long m_picoFocalPos;
   
   double m_fwFocalStepSize {0.0001};
   double m_picoFocalStepSize {100};
   
   int m_focalScale {1};
   
   //Lyot Plane
   
   std::string m_fwLyotState;
   
   
   double m_fwLyotPos;
   long m_picoLyotPos;
   
   double m_fwLyotStepSize {0.0001};
   double m_picoLyotStepSize {100};
   
   int m_lyotScale {1};
   
public:
   coronAlign( QWidget * Parent = 0, 
               Qt::WindowFlags f = Qt::WindowFlags()
             );
   
   ~coronAlign();
   
   void subscribe();
                               
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void enablePicoButtons();
   void disablePicoButtons();
   
public slots:
   void updateGUI();
   
   void on_button_pupil_u_pressed();
   void on_button_pupil_d_pressed();
   void on_button_pupil_l_pressed();
   void on_button_pupil_r_pressed();
   void on_button_pupil_scale_pressed();
   
   void on_button_focal_u_pressed();
   void on_button_focal_d_pressed();
   void on_button_focal_l_pressed();
   void on_button_focal_r_pressed();
   void on_button_focal_scale_pressed();
   
   void on_button_lyot_u_pressed();
   void on_button_lyot_d_pressed();
   void on_button_lyot_l_pressed();
   void on_button_lyot_r_pressed();
   void on_button_lyot_scale_pressed();
   
private:
     
   Ui::coronAlign ui;
};

coronAlign::coronAlign( QWidget * Parent, Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   ui.button_pupil_scale->setProperty("isScaleButton", true);
   ui.button_focal_scale->setProperty("isScaleButton", true);
   ui.button_lyot_scale->setProperty("isScaleButton", true);
   
   QTimer *timer = new QTimer(this);
   connect(timer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   timer->start(250);
      
   char ss[5];
   snprintf(ss, 5, "%d", m_pupilScale);
   ui.button_pupil_scale->setText(ss);
   
   snprintf(ss, 5, "%d", m_focalScale);
   ui.button_focal_scale->setText(ss);
   
   snprintf(ss, 5, "%d", m_lyotScale);
   ui.button_lyot_scale->setText(ss);
   
   onDisconnect();
}
   
coronAlign::~coronAlign()
{
}

void coronAlign::subscribe()
{
   if(m_parent == nullptr) return;

   m_parent->addSubscriberProperty(this, "fwpupil", "filter");
   m_parent->addSubscriberProperty(this, "fwpupil", "fsm");
   
   m_parent->addSubscriberProperty(this, "fwfpm", "filter");
   m_parent->addSubscriberProperty(this, "fwfpm", "fsm");
   
   m_parent->addSubscriberProperty(this, "fwlyot", "filter");
   m_parent->addSubscriberProperty(this, "fwlyot", "fsm");
   
   m_parent->addSubscriberProperty(this, "picomotors", "picopupil_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "picofpm_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "picolyot_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "fsm");
   
   return;
}
 
void coronAlign::onConnect()
{
   ui.labelTitle->setEnabled(true);
   ui.labelPupil->setEnabled(true);
   ui.labelFocal->setEnabled(true);
   ui.labelLyot->setEnabled(true);
   
   ui.button_pupil_u->setEnabled(true);
   ui.button_pupil_d->setEnabled(true);
   ui.button_pupil_l->setEnabled(true);
   ui.button_pupil_r->setEnabled(true);
   ui.button_pupil_scale->setEnabled(true);
   
   ui.button_focal_u->setEnabled(true);
   ui.button_focal_d->setEnabled(true);
   ui.button_focal_l->setEnabled(true);
   ui.button_focal_r->setEnabled(true);
   ui.button_focal_scale->setEnabled(true);
   
   ui.button_lyot_u->setEnabled(true);
   ui.button_lyot_d->setEnabled(true);
   ui.button_lyot_l->setEnabled(true);
   ui.button_lyot_r->setEnabled(true);
   ui.button_lyot_scale->setEnabled(true);
   
   setWindowTitle("Coronagraph Alignment");
}

void coronAlign::onDisconnect()
{
   m_picoState = "";
   m_fwPupilState = "";
   m_fwFocalState = "";
   m_fwLyotState = "";
   
   ui.labelTitle->setEnabled(false);
   ui.labelPupil->setEnabled(false);
   ui.labelFocal->setEnabled(false);
   ui.labelLyot->setEnabled(false);
   
   ui.button_pupil_u->setEnabled(false);
   ui.button_pupil_d->setEnabled(false);
   ui.button_pupil_l->setEnabled(false);
   ui.button_pupil_r->setEnabled(false);
   ui.button_pupil_scale->setEnabled(false);
   
   ui.button_focal_u->setEnabled(false);
   ui.button_focal_d->setEnabled(false);
   ui.button_focal_l->setEnabled(false);
   ui.button_focal_r->setEnabled(false);
   ui.button_focal_scale->setEnabled(false);
   
   ui.button_lyot_u->setEnabled(false);
   ui.button_lyot_d->setEnabled(false);
   ui.button_lyot_l->setEnabled(false);
   ui.button_lyot_r->setEnabled(false);
   ui.button_lyot_scale->setEnabled(false);
   
   setWindowTitle("Coronagraph Alignment (disconnected)");
}
   
void coronAlign::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   std::string dev = ipRecv.getDevice();
   if( dev == "picomotors" || 
       dev == "fwpupil" || 
       dev == "fwlyot" || 
       dev == "fwfpm" ) 
   {
      return handleSetProperty(ipRecv);
   }
   
   return;
}

void coronAlign::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::string dev = ipRecv.getDevice();
   
   if(dev == "fwpupil")
   {
      if(ipRecv.getName() == "filter")
      {
         if(ipRecv.find("current"))
         {
            m_fwPupilPos = ipRecv["current"].get<double>();
            return;
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_fwPupilState = ipRecv["state"].get();
            return;
         }
      }
   }
   if(dev == "fwfpm")
   {
      if(ipRecv.getName() == "filter")
      {
         if(ipRecv.find("current"))
         {
            m_fwFocalPos = ipRecv["current"].get<double>();
            return;
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_fwFocalState = ipRecv["state"].get();
            return;
         }
      }
   }
   if(dev == "fwlyot")
   {
      if(ipRecv.getName() == "filter")
      {
         if(ipRecv.find("current"))
         {
            m_fwLyotPos = ipRecv["current"].get<double>();
            return;
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_fwLyotState = ipRecv["state"].get();
            return;
         }
      }
   }
   else if(dev == "picomotors")
   {
      if(ipRecv.getName() == "picopupil_pos")
      {
         if(ipRecv.find("current"))
         {
            m_picoPupilPos = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "picofpm_pos")
      {
         if(ipRecv.find("current"))
         {
            m_picoFocalPos = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "picolyot_pos")
      {
         if(ipRecv.find("current"))
         {
            m_picoLyotPos = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_picoState = ipRecv["state"].get();
            return;
         }
      }
   }
   
   return;

}

void coronAlign::updateGUI()
{
   
   if(m_fwPupilState != "READY")
   {
      ui.button_pupil_l->setEnabled(false);
      ui.button_pupil_r->setEnabled(false);
      ui.button_pupil_scale->setEnabled(false);
      ui.labelPupil->setEnabled(false);
   }
   else
   {
      ui.button_pupil_l->setEnabled(true);
      ui.button_pupil_r->setEnabled(true);
      ui.button_pupil_scale->setEnabled(true);
      ui.labelPupil->setEnabled(true);
   }
   
   if(m_fwFocalState != "READY")
   {
      ui.button_focal_l->setEnabled(false);
      ui.button_focal_r->setEnabled(false);
      ui.button_focal_scale->setEnabled(false);
      ui.labelFocal->setEnabled(false);
   }
   else
   {
      ui.button_focal_l->setEnabled(true);
      ui.button_focal_r->setEnabled(true);
      ui.button_focal_scale->setEnabled(true);
      ui.labelFocal->setEnabled(true);
   }
   
   if(m_fwLyotState != "READY")
   {
      ui.button_lyot_u->setEnabled(false);
      ui.button_lyot_d->setEnabled(false);
      ui.button_lyot_scale->setEnabled(false);
      ui.labelLyot->setEnabled(false);
   }
   else
   {
      ui.button_lyot_u->setEnabled(true);
      ui.button_lyot_d->setEnabled(true);
      ui.button_lyot_scale->setEnabled(true);
      ui.labelLyot->setEnabled(true);
   }
   
   if(m_picoState != "READY")
   {
      disablePicoButtons();
   }
   else
   {
      enablePicoButtons();
   }
   

} //updateGUI()

void coronAlign::enablePicoButtons()
{   
   ui.button_pupil_u->setEnabled(true);
   ui.button_pupil_d->setEnabled(true);
   ui.button_focal_u->setEnabled(true);
   ui.button_focal_d->setEnabled(true);
   ui.button_lyot_l->setEnabled(true);
   ui.button_lyot_r->setEnabled(true);
}

void coronAlign::disablePicoButtons()
{
   ui.button_pupil_u->setEnabled(false);
   ui.button_pupil_d->setEnabled(false);
   ui.button_focal_u->setEnabled(false);
   ui.button_focal_d->setEnabled(false);
   ui.button_lyot_l->setEnabled(false);
   ui.button_lyot_r->setEnabled(false);
}

void coronAlign::on_button_pupil_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("picomotors");
   ip.setName("picopupil_pos");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_picoPupilPos + m_pupilScale*m_picoPupilStepSize;

   disablePicoButtons();
   
   sendNewProperty(ip);
}

void coronAlign::on_button_pupil_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("picomotors");
   ip.setName("picopupil_pos");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_picoPupilPos - m_pupilScale*m_picoPupilStepSize;
   
   disablePicoButtons();
   sendNewProperty(ip);
}

void coronAlign::on_button_pupil_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwpupil");
   ip.setName("filter");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_fwPupilPos + m_pupilScale*m_fwPupilStepSize;
   
   ui.button_pupil_l->setEnabled(false);
   ui.button_pupil_r->setEnabled(false);
   sendNewProperty(ip);
}

void coronAlign::on_button_pupil_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwpupil");
   ip.setName("filter");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_fwPupilPos - m_pupilScale*m_fwPupilStepSize;
   
   ui.button_pupil_l->setEnabled(false);
   ui.button_pupil_r->setEnabled(false);
   sendNewProperty(ip);
   
}

void coronAlign::on_button_pupil_scale_pressed()
{
   if( m_pupilScale == 100)
   {
      m_pupilScale = 1;
   }
   else if(m_pupilScale == 10)
   {
      m_pupilScale = 100;
   }
   else if(m_pupilScale == 1)
   {
      m_pupilScale = 10;
   }
   else
   {
      m_pupilScale = 1;
   }
   
   char ss[5];
   snprintf(ss, 5, "%d", m_pupilScale);
   std::cerr << m_pupilScale << " " << ss << "\n";
   ui.button_pupil_scale->setText(ss);


}


void coronAlign::on_button_focal_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("picomotors");
   ip.setName("picofpm_pos");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_picoFocalPos + m_focalScale*m_picoFocalStepSize;

   disablePicoButtons();
   
   sendNewProperty(ip);
   
}

void coronAlign::on_button_focal_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("picomotors");
   ip.setName("picofpm_pos");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_picoFocalPos - m_focalScale*m_picoFocalStepSize;

   disablePicoButtons();
   
   sendNewProperty(ip);
   
}

void coronAlign::on_button_focal_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwfpm");
   ip.setName("filter");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_fwFocalPos + m_focalScale*m_fwFocalStepSize;
   
   ui.button_focal_l->setEnabled(false);
   ui.button_focal_r->setEnabled(false);
   sendNewProperty(ip);
}

void coronAlign::on_button_focal_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwfpm");
   ip.setName("filter");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_fwFocalPos - m_focalScale*m_fwFocalStepSize;
   
   ui.button_focal_l->setEnabled(false);
   ui.button_focal_r->setEnabled(false);
   sendNewProperty(ip);
}

void coronAlign::on_button_focal_scale_pressed()
{
   if( m_focalScale == 100)
   {
      m_focalScale = 1;
   }
   else if(m_focalScale == 10)
   {
      m_focalScale = 100;
   }
   else if(m_focalScale == 1)
   {
      m_focalScale = 10;
   }
   else
   {
      m_focalScale = 1;
   }
   
   char ss[5];
   snprintf(ss, 5, "%d", m_focalScale);
   ui.button_focal_scale->setText(ss);


}


void coronAlign::on_button_lyot_u_pressed()
{
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwlyot");
   ip.setName("filter");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_fwLyotPos - m_lyotScale*m_fwLyotStepSize;
   
   ui.button_lyot_u->setEnabled(false);
   ui.button_lyot_d->setEnabled(false);
   sendNewProperty(ip);

}

void coronAlign::on_button_lyot_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwlyot");
   ip.setName("filter");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_fwLyotPos + m_lyotScale*m_fwLyotStepSize;
   
   ui.button_lyot_u->setEnabled(false);
   ui.button_lyot_d->setEnabled(false);
   sendNewProperty(ip);
}

void coronAlign::on_button_lyot_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("picomotors");
   ip.setName("picolyot_pos");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_picoLyotPos - m_lyotScale*m_picoLyotStepSize;

   disablePicoButtons();
   
   sendNewProperty(ip);
   
  
}

void coronAlign::on_button_lyot_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("picomotors");
   ip.setName("picolyot_pos");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_picoLyotPos + m_lyotScale*m_picoLyotStepSize;

   disablePicoButtons();
   
   sendNewProperty(ip);
   
}

void coronAlign::on_button_lyot_scale_pressed()
{
   if( m_lyotScale == 100)
   {
      m_lyotScale = 1;
   }
   else if(m_lyotScale == 10)
   {
      m_lyotScale = 100;
   }
   else if(m_lyotScale == 1)
   {
      m_lyotScale = 10;
   }
   else
   {
      m_lyotScale = 1;
   }
   
   char ss[5];
   snprintf(ss, 5, "%d", m_lyotScale);
   ui.button_lyot_scale->setText(ss);


}

} //namespace xqt
   
#include "moc_coronAlign.cpp"

#endif
