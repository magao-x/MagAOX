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

   enum camera {FLOWFS, LLOWFS, CAMSCIS};

protected:
   QMutex m_mutex;

   int m_camera {CAMSCIS};

   //Pico Motors
   std::string m_picoState;

   //Pupil Plane

   std::string m_fwPupilState;


   double m_fwPupilPos;
   long m_picoPupilPos;

   double m_fwPupilStepSize {0.0001};
   double m_picoPupilStepSize {100};

   int m_pupilScale {100};

   //Focal Plane

   std::string m_fwFocalState;


   double m_fwFocalPos;
   long m_picoFocalPos;

   double m_fwFocalStepSize {0.0001};
   double m_picoFocalStepSize {100};

   int m_focalScale {100};

   //Lyot Plane

   std::string m_fwLyotState;


   double m_fwLyotPos;
   long m_picoLyotPos;

   double m_fwLyotStepSize {0.0001};
   double m_picoLyotStepSize {100};

   int m_lyotScale {100};

    //PIAA
    std::string m_piaaState;
    double m_piaaPos;
    int m_piaaScale {1};
    double m_piaaStepSize {0.1};

    //PIAA0;
    long m_piaa0xPos;
    int m_piaa0Scale {1};
    double m_piaa0StepSize {20};

    //PIAA1;
    long m_piaa1xPos;
    long m_piaa1yPos;
    int m_piaa1Scale {1};
    double m_piaa1StepSize {20};

    //iPIAA
    std::string m_ipiaaState;
    double m_ipiaaPos;
    int m_ipiaaScale {1};
    double m_ipiaaStepSize {0.1};

    //iPIAA0;
    long m_ipiaa0xPos;
    int m_ipiaa0Scale {1};
    double m_ipiaa0StepSize {20};

    //iPIAA1;
    long m_ipiaa1xPos;
    long m_ipiaa1yPos;
    int m_ipiaa1Scale {1};
    double m_ipiaa1StepSize {20};

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

   void on_checkCamflowfs_clicked()
   {
      ui.checkCamflowfs->setCheckState(Qt::Checked);
      ui.checkCamllowfs->setCheckState(Qt::Unchecked);
      ui.checkCamsci12->setCheckState(Qt::Unchecked);
      m_camera = FLOWFS;
   }

   void on_checkCamllowfs_clicked()
   {
      ui.checkCamllowfs->setCheckState(Qt::Unchecked);
   }

   void on_checkCamsci12_clicked()
   {
      ui.checkCamflowfs->setCheckState(Qt::Unchecked);
      ui.checkCamllowfs->setCheckState(Qt::Unchecked);
      ui.checkCamsci12->setCheckState(Qt::Checked);
      m_camera = CAMSCIS;
   }

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

   void on_button_piaa_u_pressed();
   void on_button_piaa_d_pressed();
   void on_button_piaa_scale_pressed();

   void on_button_piaa0_l_pressed();
   void on_button_piaa0_r_pressed();
   void on_button_piaa0_scale_pressed();

   void on_button_piaa1_l_pressed();
   void on_button_piaa1_r_pressed();
   void on_button_piaa1_u_pressed();
   void on_button_piaa1_d_pressed();
   void on_button_piaa1_scale_pressed();

   void on_button_ipiaa_u_pressed();
   void on_button_ipiaa_d_pressed();
   void on_button_ipiaa_scale_pressed();

   void on_button_ipiaa0_l_pressed();
   void on_button_ipiaa0_r_pressed();
   void on_button_ipiaa0_scale_pressed();

   void on_button_ipiaa1_l_pressed();
   void on_button_ipiaa1_r_pressed();
   void on_button_ipiaa1_u_pressed();
   void on_button_ipiaa1_d_pressed();
   void on_button_ipiaa1_scale_pressed();

private:

   Ui::coronAlign ui;
};

coronAlign::coronAlign( QWidget * Parent, Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   ui.button_pupil_scale->setProperty("isScaleButton", true);
   ui.button_focal_scale->setProperty("isScaleButton", true);
   ui.button_lyot_scale->setProperty("isScaleButton", true);
   ui.button_piaa_scale->setProperty("isScaleButton", true);
   ui.button_piaa0_scale->setProperty("isScaleButton", true);
   ui.button_piaa1_scale->setProperty("isScaleButton", true);
   ui.button_ipiaa_scale->setProperty("isScaleButton", true);
   ui.button_ipiaa0_scale->setProperty("isScaleButton", true);
   ui.button_ipiaa1_scale->setProperty("isScaleButton", true);

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

   snprintf(ss, 5, "%d", m_piaaScale);
   ui.button_piaa_scale->setText(ss);

   snprintf(ss, 5, "%d", m_piaa0Scale);
   ui.button_piaa0_scale->setText(ss);

   snprintf(ss, 5, "%d", m_piaa1Scale);
   ui.button_piaa1_scale->setText(ss);

   snprintf(ss, 5, "%d", m_ipiaaScale);
   ui.button_ipiaa_scale->setText(ss);

   snprintf(ss, 5, "%d", m_ipiaa0Scale);
   ui.button_ipiaa0_scale->setText(ss);

   snprintf(ss, 5, "%d", m_ipiaa1Scale);
   ui.button_ipiaa1_scale->setText(ss);

   ui.checkCamllowfs->setEnabled(false);

   ui.fwpupil->setup("fwpupil");
   ui.fwfpm->setup("fwfpm");
   ui.fwlyot->setup("fwlyot");
   ui.stagepiaa->setup("stagepiaa");
   ui.stageipiaa->setup("stageipiaa");

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
   m_parent->addSubscriberProperty(this, "picomotors", "piaa0x_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "piaa1x_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "piaa1y_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "ipiaa0x_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "ipiaa1x_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "ipiaa1y_pos");
   m_parent->addSubscriberProperty(this, "picomotors", "fsm");

   m_parent->addSubscriberProperty(this, "stagepiaa", "position");
   m_parent->addSubscriberProperty(this, "stagepiaa", "fsm");

   m_parent->addSubscriberProperty(this, "stageipiaa", "position");
   m_parent->addSubscriberProperty(this, "stageipiaa", "fsm");

   m_parent->addSubscriber(ui.fwpupil);
   m_parent->addSubscriber(ui.fwfpm);
   m_parent->addSubscriber(ui.fwlyot);
   m_parent->addSubscriber(ui.stagepiaa);
   m_parent->addSubscriber(ui.stageipiaa);

   return;
}

void coronAlign::onConnect()
{
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

   ui.labelPIAA->setEnabled(true);
   ui.button_piaa_u->setEnabled(true);
   ui.button_piaa_d->setEnabled(true);
   ui.button_piaa_scale->setEnabled(true);

   ui.labelPIAA0->setEnabled(true);
   ui.button_piaa0_l->setEnabled(true);
   ui.button_piaa0_r->setEnabled(true);
   ui.button_piaa0_scale->setEnabled(true);

   ui.labelPIAA1->setEnabled(true);
   ui.button_piaa1_l->setEnabled(true);
   ui.button_piaa1_r->setEnabled(true);
   ui.button_piaa1_u->setEnabled(true);
   ui.button_piaa1_d->setEnabled(true);
   ui.button_piaa1_scale->setEnabled(true);

   ui.labeliPIAA->setEnabled(true);
   ui.button_ipiaa_u->setEnabled(true);
   ui.button_ipiaa_d->setEnabled(true);
   ui.button_ipiaa_scale->setEnabled(true);

   ui.labeliPIAA0->setEnabled(true);
   ui.button_ipiaa0_l->setEnabled(true);
   ui.button_ipiaa0_r->setEnabled(true);
   ui.button_ipiaa0_scale->setEnabled(true);

   ui.labeliPIAA1->setEnabled(true);
   ui.button_ipiaa1_l->setEnabled(true);
   ui.button_ipiaa1_r->setEnabled(true);
   ui.button_ipiaa1_u->setEnabled(true);
   ui.button_ipiaa1_d->setEnabled(true);
   ui.button_ipiaa1_scale->setEnabled(true);


   ui.fwpupil->setEnabled(true);
   ui.fwpupil->onConnect();

   ui.fwfpm->setEnabled(true);
   ui.fwfpm->onConnect();

   ui.fwlyot->setEnabled(true);
   ui.fwlyot->onConnect();

   ui.stagepiaa->setEnabled(true);
   ui.stagepiaa->onConnect();

   ui.stageipiaa->setEnabled(true);
   ui.stageipiaa->onConnect();

   setWindowTitle("Coronagraph Alignment");
}

void coronAlign::onDisconnect()
{
   m_picoState = "";
   m_fwPupilState = "";
   m_fwFocalState = "";
   m_fwLyotState = "";
   m_piaaState = "";
   m_ipiaaState = "";

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

   ui.labelPIAA->setEnabled(false);
   ui.button_piaa_u->setEnabled(false);
   ui.button_piaa_d->setEnabled(false);
   ui.button_piaa_scale->setEnabled(false);

   ui.labelPIAA0->setEnabled(false);
   ui.button_piaa0_l->setEnabled(false);
   ui.button_piaa0_r->setEnabled(false);
   ui.button_piaa0_scale->setEnabled(false);

   ui.labelPIAA1->setEnabled(false);
   ui.button_piaa1_l->setEnabled(false);
   ui.button_piaa1_r->setEnabled(false);
   ui.button_piaa1_u->setEnabled(false);
   ui.button_piaa1_d->setEnabled(false);
   ui.button_piaa1_scale->setEnabled(false);

   ui.labeliPIAA->setEnabled(false);
   ui.button_ipiaa_u->setEnabled(false);
   ui.button_ipiaa_d->setEnabled(false);
   ui.button_ipiaa_scale->setEnabled(false);

   ui.labeliPIAA0->setEnabled(false);
   ui.button_ipiaa0_l->setEnabled(false);
   ui.button_ipiaa0_r->setEnabled(false);
   ui.button_ipiaa0_scale->setEnabled(false);

   ui.labeliPIAA1->setEnabled(false);
   ui.button_ipiaa1_l->setEnabled(false);
   ui.button_ipiaa1_r->setEnabled(false);
   ui.button_ipiaa1_u->setEnabled(false);
   ui.button_ipiaa1_d->setEnabled(false);
   ui.button_ipiaa1_scale->setEnabled(false);

   ui.fwpupil->setEnabled(false);
   ui.fwpupil->onDisconnect();

   ui.fwfpm->setEnabled(true);
   ui.fwfpm->onDisconnect();

   ui.fwlyot->setEnabled(true);
   ui.fwlyot->onDisconnect();

   ui.stagepiaa->setEnabled(true);
   ui.stagepiaa->onDisconnect();

   ui.stageipiaa->setEnabled(true);
   ui.stageipiaa->onDisconnect();

   setWindowTitle("Coronagraph Alignment (disconnected)");
}

void coronAlign::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   std::string dev = ipRecv.getDevice();
   if( dev == "picomotors" ||
       dev == "fwpupil" ||
       dev == "fwlyot" ||
       dev == "fwfpm" ||
       dev == "stagepiaa" ||
       dev == "stageipiaa"
     )
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
   else if(dev == "fwfpm")
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
   else if(dev == "fwlyot")
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
   else if(dev == "stagepiaa")
   {
      if(ipRecv.getName() == "position")
      {
         if(ipRecv.find("current"))
         {
            m_piaaPos = ipRecv["current"].get<double>();
            return;
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_piaaState = ipRecv["state"].get();
            return;
         }
      }
   }
   else if(dev == "stageipiaa")
   {
      if(ipRecv.getName() == "position")
      {
         if(ipRecv.find("current"))
         {
            m_ipiaaPos = ipRecv["current"].get<double>();
            return;
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_ipiaaState = ipRecv["state"].get();
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
      else if(ipRecv.getName() == "piaa0x_pos")
      {
         if(ipRecv.find("current"))
         {
            m_piaa0xPos  = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "piaa1x_pos")
      {
         if(ipRecv.find("current"))
         {
            m_piaa1xPos  = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "piaa1y_pos")
      {
         if(ipRecv.find("current"))
         {
            m_piaa1yPos  = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "ipiaa0x_pos")
      {
         if(ipRecv.find("current"))
         {
            m_ipiaa0xPos  = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "ipiaa1x_pos")
      {
         if(ipRecv.find("current"))
         {
            m_ipiaa1xPos  = ipRecv["current"].get<long>();
            return;
         }
      }
      else if(ipRecv.getName() == "ipiaa1y_pos")
      {
         if(ipRecv.find("current"))
         {
            m_ipiaa1yPos  = ipRecv["current"].get<long>();
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
      if(m_camera == FLOWFS)
      {
         ui.button_pupil_l->setEnabled(false);
         ui.button_pupil_r->setEnabled(false);
         ui.button_pupil_scale->setEnabled(false);
         ui.labelPupil->setEnabled(false);
      }
      else
      {
         ui.button_pupil_u->setEnabled(false);
         ui.button_pupil_d->setEnabled(false);
         ui.button_pupil_scale->setEnabled(false);
         ui.labelPupil->setEnabled(false);
      }
   }
   else
   {
      if(m_camera == FLOWFS)
      {
         ui.button_pupil_l->setEnabled(true);
         ui.button_pupil_r->setEnabled(true);
         ui.button_pupil_scale->setEnabled(true);
         ui.labelPupil->setEnabled(true);
      }
      else
      {
         ui.button_pupil_u->setEnabled(true);
         ui.button_pupil_d->setEnabled(true);
         ui.button_pupil_scale->setEnabled(true);
         ui.labelPupil->setEnabled(true);
      }
   }

   if(m_fwFocalState != "READY")
   {
      if(m_camera == FLOWFS)
      {
         ui.button_focal_l->setEnabled(false);
         ui.button_focal_r->setEnabled(false);
         ui.button_focal_scale->setEnabled(false);
         ui.labelFocal->setEnabled(false);
      }
      else
      {
         ui.button_focal_u->setEnabled(false);
         ui.button_focal_d->setEnabled(false);
         ui.button_focal_scale->setEnabled(false);
         ui.labelFocal->setEnabled(false);
      }
   }
   else
   {
      if(m_camera == FLOWFS)
      {
         ui.button_focal_l->setEnabled(true);
         ui.button_focal_r->setEnabled(true);
         ui.button_focal_scale->setEnabled(true);
         ui.labelFocal->setEnabled(true);
      }
      else
      {
         ui.button_focal_u->setEnabled(true);
         ui.button_focal_d->setEnabled(true);
         ui.button_focal_scale->setEnabled(true);
         ui.labelFocal->setEnabled(true);
      }
   }

   if(m_fwLyotState != "READY")
   {
      if(m_camera == FLOWFS)
      {
         ui.button_lyot_u->setEnabled(false);
         ui.button_lyot_d->setEnabled(false);
         ui.button_lyot_scale->setEnabled(false);
         ui.labelLyot->setEnabled(false);
      }
      else
      {
         ui.button_lyot_l->setEnabled(false);
         ui.button_lyot_r->setEnabled(false);
         ui.button_lyot_scale->setEnabled(false);
         ui.labelLyot->setEnabled(false);
      }
   }
   else
   {
      if(m_camera == FLOWFS)
      {
         ui.button_lyot_u->setEnabled(true);
         ui.button_lyot_d->setEnabled(true);
         ui.button_lyot_scale->setEnabled(true);
         ui.labelLyot->setEnabled(true);
      }
      else
      {
         ui.button_lyot_l->setEnabled(true);
         ui.button_lyot_r->setEnabled(true);
         ui.button_lyot_scale->setEnabled(true);
         ui.labelLyot->setEnabled(true);
      }
   }

   if(m_piaaState != "READY")
   {
      ui.button_piaa_u->setEnabled(false);
      ui.button_piaa_d->setEnabled(false);
      ui.button_piaa_scale->setEnabled(false);
      ui.labelPIAA->setEnabled(false);
   }
   else
   {
      ui.button_piaa_u->setEnabled(true);
      ui.button_piaa_d->setEnabled(true);
      ui.button_piaa_scale->setEnabled(true);
      ui.labelPIAA->setEnabled(true);
   }

   if(m_ipiaaState != "READY")
   {
      ui.button_ipiaa_u->setEnabled(false);
      ui.button_ipiaa_d->setEnabled(false);
      ui.button_ipiaa_scale->setEnabled(false);
      ui.labeliPIAA->setEnabled(false);
   }
   else
   {
      ui.button_ipiaa_u->setEnabled(true);
      ui.button_ipiaa_d->setEnabled(true);
      ui.button_ipiaa_scale->setEnabled(true);
      ui.labeliPIAA->setEnabled(true);
   }

   if(m_picoState != "READY")
   {
      disablePicoButtons();
   }
   else
   {
      enablePicoButtons();
   }


   if(m_camera == FLOWFS)
   {
      ui.checkCamflowfs->setCheckState(Qt::Checked);
      ui.checkCamllowfs->setCheckState(Qt::Unchecked);
      ui.checkCamsci12->setCheckState(Qt::Unchecked);
   }
   else if(m_camera == LLOWFS)
   {
      ui.checkCamflowfs->setCheckState(Qt::Unchecked);
      ui.checkCamllowfs->setCheckState(Qt::Checked);
      ui.checkCamsci12->setCheckState(Qt::Unchecked);
   }
   else
   {
      ui.checkCamflowfs->setCheckState(Qt::Unchecked);
      ui.checkCamllowfs->setCheckState(Qt::Unchecked);
      ui.checkCamsci12->setCheckState(Qt::Checked);
   }

   ui.fwpupil->updateGUI();

} //updateGUI()

void coronAlign::enablePicoButtons()
{
   if(m_camera == FLOWFS)
   {
      ui.button_pupil_u->setEnabled(true);
      ui.button_pupil_d->setEnabled(true);
      ui.button_focal_u->setEnabled(true);
      ui.button_focal_d->setEnabled(true);
      ui.button_lyot_l->setEnabled(true);
      ui.button_lyot_r->setEnabled(true);
   }
   else
   {
      ui.button_pupil_l->setEnabled(true);
      ui.button_pupil_r->setEnabled(true);
      ui.button_focal_l->setEnabled(true);
      ui.button_focal_r->setEnabled(true);
      ui.button_lyot_u->setEnabled(true);
      ui.button_lyot_d->setEnabled(true);
   }

   ui.button_piaa0_l->setEnabled(true);
   ui.button_piaa0_r->setEnabled(true);
   ui.button_piaa0_scale->setEnabled(true);

   ui.button_piaa1_l->setEnabled(true);
   ui.button_piaa1_r->setEnabled(true);
   ui.button_piaa1_u->setEnabled(true);
   ui.button_piaa1_d->setEnabled(true);
   ui.button_piaa1_scale->setEnabled(true);

   ui.button_ipiaa0_l->setEnabled(true);
   ui.button_ipiaa0_r->setEnabled(true);
   ui.button_ipiaa0_scale->setEnabled(true);

   ui.button_ipiaa1_l->setEnabled(true);
   ui.button_ipiaa1_r->setEnabled(true);
   ui.button_ipiaa1_u->setEnabled(true);
   ui.button_ipiaa1_d->setEnabled(true);
   ui.button_ipiaa1_scale->setEnabled(true);
}

void coronAlign::disablePicoButtons()
{
   if(m_camera == FLOWFS)
   {
      ui.button_pupil_u->setEnabled(false);
      ui.button_pupil_d->setEnabled(false);
      ui.button_focal_u->setEnabled(false);
      ui.button_focal_d->setEnabled(false);
      ui.button_lyot_l->setEnabled(false);
      ui.button_lyot_r->setEnabled(false);
   }
   else
   {
      ui.button_pupil_l->setEnabled(false);
      ui.button_pupil_r->setEnabled(false);
      ui.button_focal_l->setEnabled(false);
      ui.button_focal_r->setEnabled(false);
      ui.button_lyot_u->setEnabled(false);
      ui.button_lyot_d->setEnabled(false);
   }

   ui.button_piaa0_l->setEnabled(false);
   ui.button_piaa0_r->setEnabled(false);
   ui.button_piaa0_scale->setEnabled(false);

   ui.button_piaa1_l->setEnabled(false);
   ui.button_piaa1_r->setEnabled(false);
   ui.button_piaa1_u->setEnabled(false);
   ui.button_piaa1_d->setEnabled(false);
   ui.button_piaa1_scale->setEnabled(false);

   ui.button_ipiaa0_l->setEnabled(false);
   ui.button_ipiaa0_r->setEnabled(false);
   ui.button_ipiaa0_scale->setEnabled(false);

   ui.button_ipiaa1_l->setEnabled(false);
   ui.button_ipiaa1_r->setEnabled(false);
   ui.button_ipiaa1_u->setEnabled(false);
   ui.button_ipiaa1_d->setEnabled(false);
   ui.button_ipiaa1_scale->setEnabled(false);
}

void coronAlign::on_button_pupil_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("picomotors");
      ip.setName("picopupil_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoPupilPos + m_pupilScale*m_picoPupilStepSize;

      disablePicoButtons();
   }
   else
   {
      ip.setDevice("fwpupil");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwPupilPos + m_pupilScale*m_fwPupilStepSize;

      ui.button_pupil_l->setEnabled(false);
      ui.button_pupil_r->setEnabled(false);
   }

   sendNewProperty(ip);
}

void coronAlign::on_button_pupil_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("picomotors");
      ip.setName("picopupil_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoPupilPos - m_pupilScale*m_picoPupilStepSize;

      disablePicoButtons();
   }
   else
   {
      ip.setDevice("fwpupil");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwPupilPos - m_pupilScale*m_fwPupilStepSize;

      ui.button_pupil_l->setEnabled(false);
      ui.button_pupil_r->setEnabled(false);
   }
   sendNewProperty(ip);
}

void coronAlign::on_button_pupil_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("fwpupil");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwPupilPos + m_pupilScale*m_fwPupilStepSize;

      ui.button_pupil_l->setEnabled(false);
      ui.button_pupil_r->setEnabled(false);
   }
   else
   {
      ip.setDevice("picomotors");
      ip.setName("picopupil_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoPupilPos + m_pupilScale*m_picoPupilStepSize;

      disablePicoButtons();
   }

   sendNewProperty(ip);
}

void coronAlign::on_button_pupil_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("fwpupil");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwPupilPos - m_pupilScale*m_fwPupilStepSize;

      ui.button_pupil_l->setEnabled(false);
      ui.button_pupil_r->setEnabled(false);
   }
   else
   {
      ip.setDevice("picomotors");
      ip.setName("picopupil_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoPupilPos - m_pupilScale*m_picoPupilStepSize;

      disablePicoButtons();
   }

   sendNewProperty(ip);

}

void coronAlign::on_button_pupil_scale_pressed()
{
   if( m_pupilScale == 100)
   {
      m_pupilScale = 10;
   }
   else if(m_pupilScale == 10)
   {
      m_pupilScale = 5;
   }
   else if(m_pupilScale == 5)
   {
      m_pupilScale = 1;
   }
   else if(m_pupilScale == 1)
   {
      m_pupilScale = 100;
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

   if(m_camera == FLOWFS)
   {
      ip.setDevice("picomotors");
      ip.setName("picofpm_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoFocalPos + m_focalScale*m_picoFocalStepSize;
   }
   else
   {
      ip.setDevice("fwfpm");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwFocalPos - m_focalScale*m_fwFocalStepSize;
      ui.button_focal_l->setEnabled(false);
      ui.button_focal_r->setEnabled(false);
   }
   disablePicoButtons();

   sendNewProperty(ip);

}

void coronAlign::on_button_focal_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("picomotors");
      ip.setName("picofpm_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoFocalPos - m_focalScale*m_picoFocalStepSize;
   }
   else
   {
      ip.setDevice("fwfpm");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwFocalPos + m_focalScale*m_fwFocalStepSize;
      ui.button_focal_l->setEnabled(false);
      ui.button_focal_r->setEnabled(false);
   }
   disablePicoButtons();

   sendNewProperty(ip);

}

void coronAlign::on_button_focal_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("fwfpm");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwFocalPos + m_focalScale*m_fwFocalStepSize;
      ui.button_focal_l->setEnabled(false);
      ui.button_focal_r->setEnabled(false);
   }
   else
   {
      ip.setDevice("picomotors");
      ip.setName("picofpm_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoFocalPos - m_focalScale*m_picoFocalStepSize;
   }

   sendNewProperty(ip);
}

void coronAlign::on_button_focal_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS)
   {
      ip.setDevice("fwfpm");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwFocalPos - m_focalScale*m_fwFocalStepSize;
      ui.button_focal_l->setEnabled(false);
      ui.button_focal_r->setEnabled(false);
   }
   else
   {
      ip.setDevice("picomotors");
      ip.setName("picofpm_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoFocalPos + m_focalScale*m_picoFocalStepSize;
   }
   sendNewProperty(ip);
}

void coronAlign::on_button_focal_scale_pressed()
{
   if( m_focalScale == 100)
   {
      m_focalScale = 10;
   }
   else if(m_focalScale == 10)
   {
      m_focalScale = 5;
   }
   else if(m_focalScale == 5)
   {
      m_focalScale = 1;
   }
   else if(m_focalScale == 1)
   {
      m_focalScale = 100;
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

   if(m_camera == FLOWFS || m_camera == CAMSCIS)
   {
      ip.setDevice("fwlyot");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwLyotPos - m_lyotScale*m_fwLyotStepSize;

      ui.button_lyot_u->setEnabled(false);
      ui.button_lyot_d->setEnabled(false);
   }
   else
   {
      ip.setDevice("picomotors");
      ip.setName("picolyot_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoLyotPos - m_lyotScale*m_picoLyotStepSize;

      disablePicoButtons();
   }

   sendNewProperty(ip);

}

void coronAlign::on_button_lyot_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS || m_camera == CAMSCIS)
   {
      ip.setDevice("fwlyot");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwLyotPos + m_lyotScale*m_fwLyotStepSize;

      ui.button_lyot_u->setEnabled(false);
      ui.button_lyot_d->setEnabled(false);
   }
   else
   {
      ip.setDevice("picomotors");
      ip.setName("picolyot_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoLyotPos + m_lyotScale*m_picoLyotStepSize;

      disablePicoButtons();
   }

   sendNewProperty(ip);

}

void coronAlign::on_button_lyot_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS || m_camera == CAMSCIS)
   {
      ip.setDevice("picomotors");
      ip.setName("picolyot_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoLyotPos + m_lyotScale*m_picoLyotStepSize;

      disablePicoButtons();
   }
   else
   {
      ip.setDevice("fwlyot");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwLyotPos - m_lyotScale*m_fwLyotStepSize;

      ui.button_lyot_u->setEnabled(false);
      ui.button_lyot_d->setEnabled(false);
   }

   sendNewProperty(ip);


}

void coronAlign::on_button_lyot_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);

   if(m_camera == FLOWFS || m_camera == CAMSCIS)
   {
      ip.setDevice("picomotors");
      ip.setName("picolyot_pos");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_picoLyotPos - m_lyotScale*m_picoLyotStepSize;

      disablePicoButtons();
   }
   else
   {
      ip.setDevice("fwlyot");
      ip.setName("filter");
      ip.add(pcf::IndiElement("target"));
      ip["target"] = m_fwLyotPos + m_lyotScale*m_fwLyotStepSize;

      ui.button_lyot_u->setEnabled(false);
      ui.button_lyot_d->setEnabled(false);
   }

   sendNewProperty(ip);

}

void coronAlign::on_button_lyot_scale_pressed()
{
   if( m_lyotScale == 100)
   {
      m_lyotScale = 10;
   }
   else if(m_lyotScale == 10)
   {
      m_lyotScale = 5;
   }
   else if(m_lyotScale == 5)
   {
      m_lyotScale = 1;
   }
   else if(m_lyotScale == 1)
   {
      m_lyotScale = 100;
   }
   else
   {
      m_lyotScale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_lyotScale);
   ui.button_lyot_scale->setText(ss);


}

void coronAlign::on_button_piaa_u_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("stagepiaa");
    ip.setName("position");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaaPos - m_piaaScale*m_piaaStepSize;

    sendNewProperty(ip);

}

void coronAlign::on_button_piaa_d_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("stagepiaa");
    ip.setName("position");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaaPos + m_piaaScale*m_piaaStepSize;

    sendNewProperty(ip);

}

void coronAlign::on_button_piaa_scale_pressed()
{
    if( m_piaaScale == 100)
   {
      m_piaaScale = 10;
   }
   else if(m_piaaScale == 10)
   {
      m_piaaScale = 5;
   }
   else if(m_piaaScale == 5)
   {
      m_piaaScale = 1;
   }
   else if(m_piaaScale == 1)
   {
      m_piaaScale = 100;
   }
   else
   {
      m_piaaScale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_piaaScale);
   ui.button_piaa_scale->setText(ss);

}

void coronAlign::on_button_piaa0_l_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("piaa0x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaa0xPos + m_piaa0Scale*m_piaa0StepSize;

    disablePicoButtons();

    sendNewProperty(ip);

}

void coronAlign::on_button_piaa0_r_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("piaa0x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaa0xPos - m_piaa0Scale*m_piaa0StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_piaa0_scale_pressed()
{
   if( m_piaa0Scale == 100)
   {
      m_piaa0Scale = 10;
   }
   else if(m_piaa0Scale == 10)
   {
      m_piaa0Scale = 5;
   }
   else if(m_piaa0Scale == 5)
   {
      m_piaa0Scale = 1;
   }
   else if(m_piaa0Scale == 1)
   {
      m_piaa0Scale = 100;
   }
   else
   {
      m_piaa0Scale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_piaa0Scale);
   ui.button_piaa0_scale->setText(ss);

}

void coronAlign::on_button_piaa1_l_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("piaa1x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaa1xPos + m_piaa1Scale*m_piaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_piaa1_r_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("piaa1x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaa1xPos - m_piaa1Scale*m_piaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_piaa1_u_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("piaa1y_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaa1yPos + m_piaa1Scale*m_piaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_piaa1_d_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("piaa1y_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_piaa1yPos - m_piaa1Scale*m_piaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_piaa1_scale_pressed()
{
    if( m_piaa1Scale == 100)
   {
      m_piaa1Scale = 10;
   }
   else if(m_piaa1Scale == 10)
   {
      m_piaa1Scale = 5;
   }
   else if(m_piaa1Scale == 5)
   {
      m_piaa1Scale = 1;
   }
   else if(m_piaa1Scale == 1)
   {
      m_piaa1Scale = 100;
   }
   else
   {
      m_piaa1Scale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_piaa1Scale);
   ui.button_piaa1_scale->setText(ss);

}

void coronAlign::on_button_ipiaa_u_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("stageipiaa");
    ip.setName("position");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaaPos - m_ipiaaScale*m_ipiaaStepSize;
    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa_d_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("stageipiaa");
    ip.setName("position");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaaPos + m_ipiaaScale*m_ipiaaStepSize;

    sendNewProperty(ip);

}

void coronAlign::on_button_ipiaa_scale_pressed()
{
    if( m_ipiaaScale == 100)
   {
      m_ipiaaScale = 10;
   }
   else if(m_ipiaaScale == 10)
   {
      m_ipiaaScale = 5;
   }
   else if(m_ipiaaScale == 5)
   {
      m_ipiaaScale = 1;
   }
   else if(m_ipiaaScale == 1)
   {
      m_ipiaaScale = 100;
   }
   else
   {
      m_ipiaaScale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_ipiaaScale);
   ui.button_ipiaa_scale->setText(ss);

}

void coronAlign::on_button_ipiaa0_l_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("ipiaa0x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaa0xPos + m_ipiaa0Scale*m_ipiaa0StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa0_r_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("ipiaa0x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaa0xPos - m_ipiaa0Scale*m_ipiaa0StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa0_scale_pressed()
{
    if( m_ipiaa0Scale == 100)
   {
      m_ipiaa0Scale = 10;
   }
   else if(m_ipiaa0Scale == 10)
   {
      m_ipiaa0Scale = 5;
   }
   else if(m_ipiaa0Scale == 5)
   {
      m_ipiaa0Scale = 1;
   }
   else if(m_ipiaa0Scale == 1)
   {
      m_ipiaa0Scale = 100;
   }
   else
   {
      m_ipiaa0Scale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_ipiaa0Scale);
   ui.button_ipiaa0_scale->setText(ss);

}

void coronAlign::on_button_ipiaa1_l_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("ipiaa1x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaa1xPos + m_ipiaa1Scale*m_ipiaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa1_r_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("ipiaa1x_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaa1xPos - m_ipiaa1Scale*m_ipiaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa1_u_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("ipiaa1y_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaa1yPos + m_ipiaa1Scale*m_ipiaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa1_d_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);

    ip.setDevice("picomotors");
    ip.setName("ipiaa1y_pos");
    ip.add(pcf::IndiElement("target"));
    ip["target"] = m_ipiaa1yPos - m_ipiaa1Scale*m_ipiaa1StepSize;

    disablePicoButtons();

    sendNewProperty(ip);
}

void coronAlign::on_button_ipiaa1_scale_pressed()
{
    if( m_ipiaa1Scale == 100)
   {
      m_ipiaa1Scale = 10;
   }
   else if(m_ipiaa1Scale == 10)
   {
      m_ipiaa1Scale = 5;
   }
   else if(m_ipiaa1Scale == 5)
   {
      m_ipiaa1Scale = 1;
   }
   else if(m_ipiaa1Scale == 1)
   {
      m_ipiaa1Scale = 100;
   }
   else
   {
      m_ipiaa1Scale = 1;
   }

   char ss[5];
   snprintf(ss, 5, "%d", m_ipiaa1Scale);
   ui.button_ipiaa1_scale->setText(ss);

}

} //namespace xqt

#include "moc_coronAlign.cpp"

#endif
