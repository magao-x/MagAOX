#ifndef pupilGuide_hpp
#define pupilGuide_hpp

#include <cmath>
#include <unistd.h>

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_pupilGuide.h"

#include "../xWidgets/xWidget.hpp"
#include "../xWidgets/statusEntry.hpp"
#include "../xWidgets/xWidget.hpp"

#define MOVE_TTM (0)
#define MOVE_TEL (1)
#define MOVE_WOOF (2)

namespace xqt 
{
   
void wooferTipTilt( double & tip,
                    double & tilt,
                    double x,
                    double y
                  )
{
   double rot = (180.+29.0)*3.14159/180.;
   double scale = -1.0;

   tip = scale*(x * cos(rot) - y * sin(rot));
   tilt = scale*(x * sin(rot) + y * cos(rot));

}


class pupilGuide : public xWidget
{
   Q_OBJECT
   
   enum camera {FLOWFS, LLOWFS, CAMSCIS};

protected:
   
   std::string m_appState;
   
   QMutex m_mutex;
   
   // --- modttm
   std::string m_modFsmState;
   int m_modState {0};
   
   double m_modCh1 {0};
   double m_modCh2 {0};
   
   double m_modFreq {0};
   double m_modFreq_tgt{0};
   double m_camwfsFreq {0};

   double m_modRad {0};
   double m_modRad_tgt{0};
  
   float m_stepSize {0.1};

   int m_tipmovewhat {MOVE_TTM};

   // --- woofer

   double m_tilt {0}; ///< current value of tilt mode from wooferModes
   double m_tip {0}; ///< current value of tip mode from wooferModes
   double m_focus {0}; ///< current value of focus mode from wooferModes

   float m_focusStepSize {0.1};

   // --- camwfs-align
   bool m_camwfsAlignLoopState {false};
   bool m_camwfsAlignLoopWaiting {false};
   QTimer * m_camwfsAlignLoopWaitTimer {nullptr};

   // --- camwfs-fit
   std::string m_camwfsfitState;
   double m_med1 {0};
   double m_med2 {0};
   double m_med3 {0};
   double m_med4 {0};
   
   double m_x1 {0};
   double m_y1 {0};
   double m_D1 {0};
   
   double m_setx1 {0};
   double m_sety1 {0};
   double m_setD1 {0};
   
   double m_x2 {0};
   double m_y2 {0};
   double m_D2 {0};
   
   double m_setx2 {0};
   double m_sety2 {0};
   double m_setD2 {0};
   
   double m_x3 {0};
   double m_y3 {0};
   double m_D3 {0};
   
   double m_setx3 {0};
   double m_sety3 {0};
   double m_setD3 {0};
   
   double m_x4 {0};
   double m_y4 {0};
   double m_D4 {0};
   
   double m_setx4 {0};
   double m_sety4 {0};
   double m_setD4 {0};

   double m_threshold_current {0};
   double m_threshold_target {0};

   // -- camwfs-avg
   std::string m_camwfsavgState;
   unsigned m_nAverage_current {0};
   unsigned m_nAverage_target {0};

   // -- dmtweeter
   std::string m_dmtweeterState;
   bool m_dmtweeterTestSet {false};

   // -- dmncpc
   std::string m_dmncpcState;
   bool m_dmncpcTestSet {false};

   // -- ttmpupil
   std::string m_pupFsmState;
   double m_pupCh1 {0};
   double m_pupCh2 {0};
   
   float m_pupStepSize {0.1};

   int m_pupCam {CAMSCIS};

   // -- ttmperi
   std::string m_ttmPeriFsmState;
   double m_ttmPeriCh1 {0};
   double m_ttmPeriCh2 {0};

   float m_ttmPeriStepSize {0.1};


   // -- Camera Lens
   std::string m_camlensxFsmState;
   std::string m_camlensyFsmState;
   float m_camlensx_pos {0};
   float m_camlensy_pos {0};
   
   float m_camlensStepSize {0.01};
   
public:
   pupilGuide( QWidget * Parent = 0, 
               Qt::WindowFlags f = Qt::WindowFlags()
             );
   
   ~pupilGuide();
   
   void subscribe();
                               
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
   void modGUISetEnable( bool enableModGUI,
                         bool enableModArrows
                       );
   
   void camwfsfitSetEnabled(bool enabled);
   
   void camlensSetEnabled(bool enabled);
   
public slots:
   void updateGUI();
   
   //----------- modttm

   void on_buttonMod_mod_pressed();
   void on_buttonMod_set_pressed();
   void on_buttonMod_rest_pressed();

   void on_button_scalemodcamwfs_pressed();
         
   void on_button_ttmtel_pressed();

   void on_button_tip_u_pressed();
   void on_button_tip_ul_pressed();
   void on_button_tip_l_pressed();
   void on_button_tip_dl_pressed();
   void on_button_tip_d_pressed();
   void on_button_tip_dr_pressed();
   void on_button_tip_r_pressed();
   void on_button_tip_ur_pressed();
   void on_button_tip_scale_pressed();

   //------------- focus
   void on_button_focus_p_pressed();
   void on_button_focus_m_pressed();
   void on_button_focus_scale_pressed();

   //----------- dmtweeter
   void on_buttonTweeterTest_set_pressed();

   //----------- dmncpc
   void on_buttonNCPCTest_set_pressed();

   //----------- ttmpupil
   void on_buttonPup_rest_pressed();
   void on_buttonPup_set_pressed();

   void on_button_camera_pressed();

   void on_button_pup_ul_pressed();
   void on_button_pup_dl_pressed();
   void on_button_pup_dr_pressed();
   void on_button_pup_ur_pressed();
   void on_button_pup_scale_pressed();
   
   //---------- TTM Peri
   void on_button_ttmPeri_rest_pressed();
   void on_button_ttmPeri_set_pressed();

   void on_button_ttmPeri_l_pressed();
   void on_button_ttmPeri_r_pressed();
   void on_button_ttmPeri_u_pressed();
   void on_button_ttmPeri_d_pressed();
   void on_button_ttmPeri_scale_pressed();
   
   void toggleExpFit(bool visible);
   void on_buttonExpFit_pressed();

   void on_pupilAlign_loopSlider_sliderReleased();
   void on_pupilAlign_loopSlider_sliderPressed();
   void on_pupilAlign_loopSlider_sliderMoved(int);
   void camwfsAlignLoopWaitTimerOut();

   void on_button_camlens_u_pressed();
   void on_button_camlens_l_pressed();
   void on_button_camlens_d_pressed();
   void on_button_camlens_r_pressed();
   void on_button_camlens_scale_pressed();
private:
     
   Ui::pupilGuide ui;
};



pupilGuide::pupilGuide( QWidget * Parent, Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);

   ui.modState->setProperty("isStatus", true);


   ui.camlens_fsm->setProperty("isStatus", true);
   ui.button_camlens_scale->setProperty("isScaleButton", true);
   ui.button_tip_scale->setProperty("isScaleButton", true);
   ui.button_focus_scale->setProperty("isScaleButton", true);
   ui.button_pup_scale->setProperty("isScaleButton", true);
   ui.button_ttmPeri_scale->setProperty("isScaleButton", true);
   
   QTimer *timer = new QTimer(this);
   connect(timer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   timer->start(250);
      
   ui.modFreq_current->setup("modwfs", "modFrequency", statusEntry::FLOAT, "", "");
   ui.modFreq_current->setStretch(0,0,6);//removes spacer and maximizes text field
   ui.modFreq_current->format("%0.1f");
   ui.modFreq_current->onDisconnect();

   ui.modRad_current->setup("modwfs", "modRadius", statusEntry::FLOAT, "", "");
   ui.modRad_current->setStretch(0,0,6);//removes spacer and maximizes text field
   ui.modRad_current->format("%0.1f");
   ui.modRad_current->onDisconnect();

   ui.modCh1->setup("fxngenmodwfs", "C1ofst", statusEntry::FLOAT, "Ch1", "V");
   ui.modCh1->currEl("value");
   ui.modCh1->targEl("value");
   ui.modCh1->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.modCh1->format("%0.2f");
   ui.modCh1->onDisconnect();

   ui.modCh2->setup("fxngenmodwfs", "C2ofst", statusEntry::FLOAT, "Ch2", "V");
   ui.modCh2->currEl("value");
   ui.modCh2->targEl("value");
   ui.modCh2->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.modCh2->format("%0.2f");
   ui.modCh2->onDisconnect();

   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.button_tip_scale->setText(ss);
   
   snprintf(ss, 5, "%0.2f", m_focusStepSize);
   ui.button_focus_scale->setText(ss);

   snprintf(ss, 5, "%0.2f", m_pupStepSize);
   ui.button_pup_scale->setText(ss);
   
   snprintf(ss, 5, "%0.2f", m_camlensStepSize*10);
   ui.button_camlens_scale->setText(ss);
   
   setXwFont(ui.labelModAndCenter);
   setXwFont(ui.modState);
   setXwFont(ui.labelFreq);
   setXwFont(ui.labelRad);
   setXwFont(ui.buttonMod_rest);
   setXwFont(ui.buttonMod_set);
   setXwFont(ui.buttonMod_mod);
   setXwFont(ui.labelMedianFluxes);
   setXwFont(ui.med1);
   setXwFont(ui.med2);
   setXwFont(ui.med3);
   setXwFont(ui.med4);
   setXwFont(ui.setDelta);

   m_tipmovewhat = MOVE_TTM;
   on_button_ttmtel_pressed();

   //tweeter controls
   setXwFont(ui.labelTweeter);
   setXwFont(ui.buttonTweeterTest_set);

   //ncpc controls
   setXwFont(ui.labelNCPC);
   setXwFont(ui.buttonNCPCTest_set);

   //-----------ttmpupil controls ------------
   setXwFont(ui.labelPupilSteering);
   //setXwFont(ui.pupState);
   setXwFont(ui.buttonPup_rest);
   setXwFont(ui.buttonPup_set);
   ui.pupState->device("ttmpupil");
   ui.pupCh1->setup("ttmpupil", "pos_1", statusEntry::FLOAT, "Ch 1", "V");
   ui.pupCh1->setStretch(1,2,4);
   ui.pupCh1->highlightChanges(false);
   ui.pupCh1->onDisconnect();
   ui.pupCh2->setup("ttmpupil", "pos_2", statusEntry::FLOAT, "Ch 2", "V");
   ui.pupCh2->setStretch(1,2,4);
   ui.pupCh2->highlightChanges(false);
   ui.pupCh2->onDisconnect();

   //-----------ttmperi controls ------------
   setXwFont(ui.labelTTMPeri);
   //setXwFont(ui.ttmPeriState);
   setXwFont(ui.buttonPup_rest);
   setXwFont(ui.buttonPup_set);
   ui.ttmPeriState->device("ttmperi");
   ui.ttmPeriCh1->setup("ttmperi", "axis1_voltage", statusEntry::FLOAT, "Ch 1", "V");
   ui.ttmPeriCh1->setStretch(1,2,4);
   ui.ttmPeriCh1->highlightChanges(false);
   ui.ttmPeriCh1->onDisconnect();
   ui.ttmPeriCh2->setup("ttmperi", "axis2_voltage", statusEntry::FLOAT, "Ch 2", "V");
   ui.ttmPeriCh2->highlightChanges(false);
   ui.ttmPeriCh2->setStretch(1,2,4);
   ui.ttmPeriCh2->onDisconnect();





   setXwFont(ui.labelPupilAlignment);
   m_camwfsAlignLoopWaitTimer = new QTimer;
   connect(m_camwfsAlignLoopWaitTimer, SIGNAL(timeout()), this, SLOT(camwfsAlignLoopWaitTimerOut()));

   ui.pupilAlign_gain->setup("camwfs-align", "loop_gain", statusEntry::FLOAT, "loop gain", "");
   ui.pupilAlign_gain->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.pupilAlign_gain->format("%0.2f");
   ui.pupilAlign_gain->onDisconnect();


   setXwFont(ui.camlens_label);
   setXwFont(ui.camlens_fsm);

   setXwFont(ui.labelPupilFitting);//,1.2);

   setXwFont(ui.labelx);
   setXwFont(ui.labely);
   setXwFont(ui.labelD);
   setXwFont(ui.labelUR);
   setXwFont(ui.labelUL);
   setXwFont(ui.labelLR);
   setXwFont(ui.labelLL);
   setXwFont(ui.labelAvg);
   setXwFont(ui.coordUR_x);
   setXwFont(ui.coordUR_y);
   setXwFont(ui.coordUR_D);
   setXwFont(ui.coordUL_x);
   setXwFont(ui.coordUL_y);
   setXwFont(ui.coordUL_D);
   setXwFont(ui.coordLR_x);
   setXwFont(ui.coordLR_y);
   setXwFont(ui.coordLR_D);
   setXwFont(ui.coordLL_x);
   setXwFont(ui.coordLL_y);
   setXwFont(ui.coordLL_D);
   setXwFont(ui.coordAvg_x);
   setXwFont(ui.coordAvg_y);
   setXwFont(ui.coordAvg_D);

   setXwFont(ui.setDelta_pup);

   ui.fitThreshold->setup("camwfs-fit", "threshold", statusEntry::FLOAT, "Thresh", "");
   ui.fitThreshold->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.fitThreshold->format("%0.3f");
   ui.fitThreshold->onDisconnect();

   ui.fitAvgTime->setup("camwfs-avg", "avgTime", statusEntry::FLOAT, "Avg. T.", "s");
   ui.fitAvgTime->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.fitAvgTime->format("%0.3f");
   ui.fitAvgTime->onDisconnect();

   ui.camlens_x_pos->setup("stagecamlensx", "position", statusEntry::FLOAT, "X", "mm");
   ui.camlens_x_pos->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.camlens_x_pos->format("%0.4f");
   ui.camlens_x_pos->onDisconnect();

   ui.camlens_y_pos->setup("stagecamlensy", "position", statusEntry::FLOAT, "Y", "mm");
   ui.camlens_y_pos->setStretch(0,1,6);//removes spacer and maximizes text field
   ui.camlens_y_pos->format("%0.4f");
   ui.camlens_y_pos->onDisconnect();

   //Set the pupil fit boxes to invisible at startup   
   toggleExpFit(false);
}
   
pupilGuide::~pupilGuide()
{
}

void pupilGuide::subscribe()
{
   if(m_parent == nullptr) return;

   m_parent->addSubscriberProperty(this, "modwfs", "modFrequency");
   m_parent->addSubscriberProperty(this, "modwfs", "modRadius");
   m_parent->addSubscriberProperty(this, "modwfs", "modState");
   m_parent->addSubscriberProperty(this, "modwfs", "fsm");

   m_parent->addSubscriber(ui.modFreq_current);
   m_parent->addSubscriber(ui.modRad_current);
   m_parent->addSubscriber(ui.modCh1);
   m_parent->addSubscriber(ui.modCh2);
   
   m_parent->addSubscriberProperty(this, "camwfs", "fps");

   m_parent->addSubscriberProperty(this, "wooferModes", "current_amps");

   m_parent->addSubscriberProperty(this, "camwfs-fit", "fsm");
   m_parent->addSubscriberProperty(this, "camwfs-fit", "quadrant1");
   m_parent->addSubscriberProperty(this, "camwfs-fit", "quadrant2");
   m_parent->addSubscriberProperty(this, "camwfs-fit", "quadrant3");
   m_parent->addSubscriberProperty(this, "camwfs-fit", "quadrant4");
   m_parent->addSubscriberProperty(this, "camwfs-fit", "threshold");
   
   m_parent->addSubscriberProperty(this, "camwfs-avg", "fsm");
   m_parent->addSubscriberProperty(this, "camwfs-avg", "nAverage");

   
   m_parent->addSubscriber(ui.pupState);
   m_parent->addSubscriber(ui.pupCh1);
   m_parent->addSubscriber(ui.pupCh2);
   m_parent->addSubscriberProperty(this, "ttmpupil", "fsm");
   m_parent->addSubscriberProperty(this, "ttmpupil", "pos_1");
   m_parent->addSubscriberProperty(this, "ttmpupil", "pos_2");

   m_parent->addSubscriber(ui.ttmPeriState);
   m_parent->addSubscriber(ui.ttmPeriCh1);
   m_parent->addSubscriber(ui.ttmPeriCh2);
   m_parent->addSubscriberProperty(this, "ttmperi", "fsm");
   m_parent->addSubscriberProperty(this, "ttmperi", "axis_1");
   m_parent->addSubscriberProperty(this, "ttmperi", "axix_2");

   m_parent->addSubscriberProperty(this, "dmtweeter", "fsm");
   m_parent->addSubscriberProperty(this, "dmtweeter", "test_set");
   m_parent->addSubscriberProperty(this, "dmtweeter", "test");

   m_parent->addSubscriberProperty(this, "dmncpc", "fsm");
   m_parent->addSubscriberProperty(this, "dmncpc", "test_set");
   m_parent->addSubscriberProperty(this, "dmncpc", "test");


   m_parent->addSubscriberProperty(this, "camwfs-align", "fsm");
   m_parent->addSubscriberProperty(this, "camwfs-align", "loop_state");
   m_parent->addSubscriber(ui.pupilAlign_gain);

   m_parent->addSubscriber(ui.fitThreshold);
   m_parent->addSubscriber(ui.fitAvgTime);

   m_parent->addSubscriberProperty(this, "stagecamlensx", "fsm");   
   m_parent->addSubscriberProperty(this, "stagecamlensy", "fsm");
   m_parent->addSubscriberProperty(this, "stagecamlensx", "position"); //we need these too   
   m_parent->addSubscriberProperty(this, "stagecamlensy", "position");
   m_parent->addSubscriber(ui.camlens_x_pos);
   m_parent->addSubscriber(ui.camlens_y_pos);

   return;
}
   
void pupilGuide::onConnect()
{
   ui.labelModAndCenter->setEnabled(true);
   ui.labelPupilFitting->setEnabled(true);

   ui.modFreq_current->onConnect();
   ui.modRad_current->onConnect();
   ui.modCh1->onConnect();
   ui.modCh2->onConnect();

   ui.pupState->onConnect();
   ui.pupCh1->onConnect();
   ui.pupCh2->onConnect();


   ui.ttmPeriState->onConnect();
   ui.ttmPeriCh1->onConnect();
   ui.ttmPeriCh2->onConnect();

   ui.labelPupilSteering->setEnabled(true);
   
   ui.pupilAlign_gain->onConnect();
   
   ui.camlens_label->setEnabled(true);
   
   ui.fitThreshold->onConnect();
   ui.fitAvgTime->onConnect();
   ui.camlens_x_pos->onConnect();
   ui.camlens_y_pos->onConnect();

   setWindowTitle("Pupil Alignment");
}

void pupilGuide::onDisconnect()
{
   m_modFsmState = "";
   m_pupFsmState = "";

   ui.modFreq_current->onDisconnect();
   ui.modRad_current->onDisconnect();
   ui.modCh1->onDisconnect();
   ui.modCh2->onDisconnect();

   ui.pupState->onDisconnect();
   ui.pupCh1->onDisconnect();
   ui.pupCh2->onDisconnect();

   ui.ttmPeriState->onDisconnect();
   ui.ttmPeriCh1->onDisconnect();
   ui.ttmPeriCh2->onDisconnect();
   

   m_camlensxFsmState = "";
   m_camlensyFsmState = "";
   m_camwfsavgState = "";
   m_camwfsfitState = "";

   ui.labelModAndCenter->setEnabled(false);
   ui.labelPupilFitting->setEnabled(false);
   ui.labelPupilSteering->setEnabled(false);


   ui.pupilAlign_gain->onDisconnect();

   ui.camlens_label->setEnabled(false);
   
   ui.fitThreshold->onDisconnect();
   ui.fitAvgTime->onDisconnect();
   ui.camlens_x_pos->onDisconnect();
   ui.camlens_y_pos->onDisconnect();

   setWindowTitle("Pupil Alignment (disconnected)");
   
}
   
void pupilGuide::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   std::string dev = ipRecv.getDevice();
   if( dev == "modwfs" || 
       dev == "camwfs" ||
       dev == "wooferModes" ||
       dev == "camwfs-avg" || 
       dev == "camwfs-fit" || 
       /*dev == "fxngenmodwfs" || */
       dev == "ttmpupil" || 
       dev == "camwfs-align" ||
       dev == "stagecamlensx" || 
       dev == "stagecamlensy" ||
       dev == "dmtweeter" ||
       dev == "dmncpc" ||
       dev == "ttmperi" ) 
   {
      return handleSetProperty(ipRecv);
   }
   
   return;
}

void pupilGuide::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::string dev = ipRecv.getDevice();
   
   if(dev == "modwfs")
   {
      if(ipRecv.getName() == "modFrequency")
      {
         if(ipRecv.find("current"))
         {
            m_modFreq = ipRecv["current"].get<double>();
         }
         if(ipRecv.find("target"))
         {
            m_modFreq_tgt = ipRecv["target"].get<double>();
         }
      }
      else if(ipRecv.getName() == "modRadius")
      {
         if(ipRecv.find("current"))
         {
            m_modRad = ipRecv["current"].get<double>();
         }
         if(ipRecv.find("target"))
         {
            m_modRad_tgt = ipRecv["target"].get<double>();
         }
      }
      else if(ipRecv.getName() == "modState")
      {
         if(ipRecv.find("current"))
         {
            m_modState = ipRecv["current"].get<int>();
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_modFsmState = ipRecv["state"].get<std::string>();
         }
      }
   }
   else if(dev == "camwfs")
   {
      if(ipRecv.getName() == "fps")
      {
         if(ipRecv.find("current"))
         {
            m_camwfsFreq = ipRecv["current"].get<double>();
         }
      }
   }
   else if(dev == "wooferModes")
   {
      if(ipRecv.getName() == "current_amps")
      {
         if(ipRecv.find("0000"))
         {
            m_tip = ipRecv["0000"].get<double>();
         }
         if(ipRecv.find("0001"))
         {
            m_tilt = ipRecv["0001"].get<double>();
         }
         if(ipRecv.find("0002"))
         {
            m_focus = ipRecv["0002"].get<double>();
         }
      }
   }
   else if(dev == "camwfs-avg")
   {
      if(ipRecv.getName() == "nAverage")
      {
         if(ipRecv.find("current"))
         {
            m_nAverage_current = ipRecv["current"].get<unsigned>();
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_camwfsavgState = ipRecv["state"].get<std::string>();
         }
      }
   }
   else if(dev == "camwfs-fit")
   {
      
      if(ipRecv.getName() == "quadrant1")
      {
         if(ipRecv.find("med"))
         {
            m_med1 = ipRecv["med"].get<double>();
         }
         
         if(ipRecv.find("x"))
         {
            m_x1 = ipRecv["x"].get<double>();
         }
      
         if(ipRecv.find("y"))
         {
            m_y1 = ipRecv["y"].get<double>();
         }
         
         if(ipRecv.find("D"))
         {
            m_D1 = ipRecv["D"].get<double>();
         }
         
         if(ipRecv.find("set-x"))
         {
            m_setx1 = ipRecv["set-x"].get<double>();
         }
      
         if(ipRecv.find("set-y"))
         {
            m_sety1 = ipRecv["set-y"].get<double>();
         }
         
         if(ipRecv.find("set-D"))
         {
            m_setD1 = ipRecv["set-D"].get<double>();
         }
      }
      else if(ipRecv.getName() == "quadrant2")
      {
         if(ipRecv.find("med"))
         {
            m_med2 = ipRecv["med"].get<double>();
         }
         
         if(ipRecv.find("x"))
         {
            m_x2 = ipRecv["x"].get<double>();
         }
      
         if(ipRecv.find("y"))
         {
            m_y2 = ipRecv["y"].get<double>();
         }
         
         if(ipRecv.find("D"))
         {
            m_D2 = ipRecv["D"].get<double>();
         }
         
         if(ipRecv.find("set-x"))
         {
            m_setx2 = ipRecv["set-x"].get<double>();
         }
      
         if(ipRecv.find("set-y"))
         {
            m_sety2 = ipRecv["set-y"].get<double>();
         }
         
         if(ipRecv.find("set-D"))
         {
            m_setD2 = ipRecv["set-D"].get<double>();
         }
      }
      else if(ipRecv.getName() == "quadrant3")
      {
         if(ipRecv.find("med"))
         {
            m_med3 = ipRecv["med"].get<double>();
         }
         
         if(ipRecv.find("x"))
         {
            m_x3 = ipRecv["x"].get<double>();
         }
      
         if(ipRecv.find("y"))
         {
            m_y3 = ipRecv["y"].get<double>();
         }
         
         if(ipRecv.find("D"))
         {
            m_D3 = ipRecv["D"].get<double>();
         }
         
         if(ipRecv.find("set-x"))
         {
            m_setx3 = ipRecv["set-x"].get<double>();
         }
      
         if(ipRecv.find("set-y"))
         {
            m_sety3 = ipRecv["set-y"].get<double>();
         }
         
         if(ipRecv.find("set-D"))
         {
            m_setD3 = ipRecv["set-D"].get<double>();
         }
      }
      else if(ipRecv.getName() == "quadrant4")
      {
         if(ipRecv.find("med"))
         {
            m_med4 = ipRecv["med"].get<double>();
         }
         
         if(ipRecv.find("x"))
         {
            m_x4 = ipRecv["x"].get<double>();
         }
      
         if(ipRecv.find("y"))
         {
            m_y4 = ipRecv["y"].get<double>();
         }
         
         if(ipRecv.find("D"))
         {
            m_D4 = ipRecv["D"].get<double>();
         }
         
         if(ipRecv.find("set-x"))
         {
            m_setx4 = ipRecv["set-x"].get<double>();
         }
      
         if(ipRecv.find("set-y"))
         {
            m_sety4 = ipRecv["set-y"].get<double>();
         }
         
         if(ipRecv.find("set-D"))
         {
            m_setD4 = ipRecv["set-D"].get<double>();
         }
      }   
      else if(ipRecv.getName() == "threshold")
      {
         if(ipRecv.find("current"))
         {
            m_threshold_current = ipRecv["current"].get<double>();
         }
      }
      else if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_camwfsfitState = ipRecv["state"].get<std::string>();
         }
      }
      
      
   }
   else if(dev == "ttmpupil")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_pupFsmState = ipRecv["state"].get<std::string>();
         }
      }
      else if(ipRecv.getName() == "pos_1")
      {
         if(ipRecv.find("current"))
         {
            m_pupCh1 = ipRecv["current"].get<double>();
         }
      }
      else if(ipRecv.getName() == "pos_2")
      {
         if(ipRecv.find("current"))
         {
            m_pupCh2 = ipRecv["current"].get<double>();
         }
      }
   }
   else if(dev == "ttmperi")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_ttmPeriFsmState = ipRecv["state"].get<std::string>();
         }
      }
      else if(ipRecv.getName() == "axis1_voltage")
      {
         if(ipRecv.find("current"))
         {
            m_ttmPeriCh1 = ipRecv["current"].get<double>();
         }
      }
      else if(ipRecv.getName() == "axis2_voltage")
      {
         if(ipRecv.find("current"))
         {
            m_ttmPeriCh2 = ipRecv["current"].get<double>();
         }
      }
   }
   else if(dev == "camwfs-align")
   {
      if(ipRecv.getName() == "loop_state")
      {
         if(ipRecv.find("toggle"))
         {
            if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
            {
               if(m_camwfsAlignLoopState == false) m_camwfsAlignLoopWaiting = false;
               m_camwfsAlignLoopState = true;
            }
            else
            {
               if(m_camwfsAlignLoopState == true) m_camwfsAlignLoopWaiting = false;
               m_camwfsAlignLoopState = false;
            }
         
            
         }
      }
   }
   else if(dev == "stagecamlensx")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_camlensxFsmState = ipRecv["state"].get<std::string>();
         }
      }
      else if(ipRecv.getName() == "position")
      {
         if(ipRecv.find("current"))
         {
            m_camlensx_pos = ipRecv["current"].get<float>();
         }
      }
   }
   else if(dev == "stagecamlensy")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_camlensyFsmState = ipRecv["state"].get<std::string>();
         }
      }
      else if(ipRecv.getName() == "position")
      {
         if(ipRecv.find("current"))
         {
            m_camlensy_pos = ipRecv["current"].get<float>();
         }
      }
   }
   else if(dev == "dmtweeter")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_dmtweeterState = ipRecv["state"].get<std::string>();
         }
      }
      else if(ipRecv.getName() == "test_set")
      {
         if(ipRecv.find("toggle"))
         {
            if(ipRecv["toggle"] == pcf::IndiElement::On) m_dmtweeterTestSet = true;
            else m_dmtweeterTestSet=false;
         }
      }
   }
   else if(dev == "dmncpc")
   {
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_dmncpcState = ipRecv["state"].get<std::string>();
         }
      }
      else if(ipRecv.getName() == "test_set")
      {
         if(ipRecv.find("toggle"))
         {
            if(ipRecv["toggle"] == pcf::IndiElement::On) m_dmncpcTestSet = true;
            else m_dmncpcTestSet=false;
         }
      }
   }

   return;
   
}

void pupilGuide::modGUISetEnable( bool enableModGUI,
                                  bool enableModArrows
                                )
{
   if(enableModGUI)
   {
      ui.modState->setEnabled(true);
      ui.modFreq_current->setEnabled(true);
      ui.modRad_current->setEnabled(true);
      ui.modCh1->setEnabled(true);
      ui.modCh2->setEnabled(true);
      
      if(enableModArrows)
      {
         ui.button_tip_ul->setEnabled(true); 
         ui.button_tip_u->setEnabled(true);
         ui.button_tip_ur->setEnabled(true); 
         ui.button_tip_l->setEnabled(true); 
         ui.button_tip_scale->setEnabled(true);
         ui.button_tip_r->setEnabled(true);
         ui.button_tip_dl->setEnabled(true);
         ui.button_tip_d->setEnabled(true);
         ui.button_tip_dr->setEnabled(true);

         if(m_tipmovewhat == MOVE_TEL || m_tipmovewhat == MOVE_WOOF)
         {
            ui.button_focus_p->setEnabled(true);
            ui.button_focus_scale->setEnabled(true);
            ui.button_focus_m->setEnabled(true);
         }
         else
         {
            ui.button_focus_p->setEnabled(false);
            ui.button_focus_scale->setEnabled(false);
            ui.button_focus_m->setEnabled(false);
         }
      }
      else
      {
         ui.button_tip_ul->setEnabled(false); 
         ui.button_tip_u->setEnabled(false);
         ui.button_tip_ur->setEnabled(false); 
         ui.button_tip_l->setEnabled(false); 
         ui.button_tip_scale->setEnabled(false);
         ui.button_tip_r->setEnabled(false);
         ui.button_tip_dl->setEnabled(false);
         ui.button_tip_d->setEnabled(false);
         ui.button_tip_dr->setEnabled(false);

         ui.button_focus_p->setEnabled(false);
         ui.button_focus_scale->setEnabled(false);
         ui.button_focus_m->setEnabled(false);
      }
   }
   else
   {
      ui.modFreq_current->setEnabled(false);
      ui.modRad_current->setEnabled(false);
      ui.buttonMod_rest->setEnabled(false);
      ui.buttonMod_set->setEnabled(false);
      ui.buttonMod_mod->setEnabled(false);
      ui.modCh1->setEnabled(false);
      ui.modCh2->setEnabled(false);
      
      if(m_tipmovewhat == MOVE_TTM) 
      {
         ui.button_tip_ul->setEnabled(false); 
         ui.button_tip_u->setEnabled(false);
         ui.button_tip_ur->setEnabled(false); 
         ui.button_tip_l->setEnabled(false); 
         ui.button_tip_scale->setEnabled(false);
         ui.button_tip_r->setEnabled(false);
         ui.button_tip_dl->setEnabled(false);
         ui.button_tip_d->setEnabled(false);
         ui.button_tip_dr->setEnabled(false);

         ui.button_focus_p->setEnabled(false);
         ui.button_focus_scale->setEnabled(false);
         ui.button_focus_m->setEnabled(false);
      }
      else
      {
         ui.button_tip_ul->setEnabled(true); 
         ui.button_tip_u->setEnabled(true);
         ui.button_tip_ur->setEnabled(true); 
         ui.button_tip_l->setEnabled(true); 
         ui.button_tip_scale->setEnabled(true);
         ui.button_tip_r->setEnabled(true);
         ui.button_tip_dl->setEnabled(true);
         ui.button_tip_d->setEnabled(true);
         ui.button_tip_dr->setEnabled(true);

         ui.button_focus_p->setEnabled(true);
         ui.button_focus_scale->setEnabled(true);
         ui.button_focus_m->setEnabled(true);
      }
   }
}

void pupilGuide::camwfsfitSetEnabled(bool enabled)
{
   ui.labelMedianFluxes->setEnabled(enabled);
   ui.med1->setEnabled(enabled);
   ui.med2->setEnabled(enabled);
   ui.med3->setEnabled(enabled);
   ui.med4->setEnabled(enabled);
   ui.setDelta->setEnabled(enabled);
   ui.fitThreshold->setEnabled(enabled);
   
   if(enabled == false)
   {
      ui.med1->setText("");
      ui.med2->setText("");
      ui.med3->setText("");
      ui.med4->setText("");
   }
   
   ui.coordLL_D->setEnabled(enabled);
   ui.coordLR_D->setEnabled(enabled);
   ui.coordUL_D->setEnabled(enabled);
   ui.coordUR_D->setEnabled(enabled);
   ui.coordLL_x->setEnabled(enabled);
   ui.coordLR_x->setEnabled(enabled);
   ui.coordUL_x->setEnabled(enabled);
   ui.coordUR_x->setEnabled(enabled);
   ui.coordLL_y->setEnabled(enabled);
   ui.coordLR_y->setEnabled(enabled);
   ui.coordUL_y->setEnabled(enabled);
   ui.coordUR_y->setEnabled(enabled);
   ui.coordAvg_D->setEnabled(enabled);
   ui.coordAvg_x->setEnabled(enabled);
   ui.coordAvg_y->setEnabled(enabled);
   ui.setDelta_pup->setEnabled(enabled);
   ui.labelx->setEnabled(enabled);
   ui.labely->setEnabled(enabled);
   ui.labelD->setEnabled(enabled);
   ui.labelUR->setEnabled(enabled);
   ui.labelUL->setEnabled(enabled);
   ui.labelLR->setEnabled(enabled);
   ui.labelLL->setEnabled(enabled);
   ui.labelAvg->setEnabled(enabled);
}

void pupilGuide::camlensSetEnabled(bool enabled)
{
   ui.camlens_fsm->setEnabled(enabled);
   ui.camlens_x_pos->setEnabled(enabled);
   ui.camlens_y_pos->setEnabled(enabled);
   ui.button_camlens_u->setEnabled(enabled);
   ui.button_camlens_l->setEnabled(enabled);
   ui.button_camlens_r->setEnabled(enabled);
   ui.button_camlens_d->setEnabled(enabled);
   ui.button_camlens_scale->setEnabled(enabled);
}

void pupilGuide::updateGUI()
{
   
   //--------- Modulation 
   
   bool enableModGUI = true;
   bool enableModArrows = true;
   
   char str[16];
   if(m_modFsmState == "READY")
   {
      if(m_modState == 1)
      {
         ui.modState->setText("RIP");
         if(m_tipmovewhat == MOVE_TTM) enableModArrows = false;
      }
      else
      {
         ui.modState->setText("SET");
      }
   }
   else if(m_modFsmState == "OPERATING")
   {
      ui.modState->setText("MODULATING");
   }
   else
   {
      enableModGUI = false;
      if(m_modFsmState == "")
      {
         ui.modState->setText("STATE UNKNOWN");
         ui.modState->setEnabled(false);
      }
      else
      {
         ui.modState->setText(m_modFsmState.c_str());
         ui.modState->setEnabled(true);
      }
   }
   
   modGUISetEnable(enableModGUI, enableModArrows);

   ui.modRad_current->updateGUI();
   ui.modFreq_current->updateGUI();
   ui.modCh1->updateGUI();
   ui.modCh2->updateGUI();
   
   if(m_modState == 3 && enableModGUI)
   {
      ui.buttonMod_rest->setEnabled(true);
      ui.buttonMod_set->setEnabled(false);
      ui.buttonMod_mod->setEnabled(true);
   }
   else if(m_modState == 4 && enableModGUI)
   {
      ui.buttonMod_rest->setEnabled(true);
      ui.buttonMod_set->setEnabled(true);
      ui.buttonMod_mod->setEnabled(true);
   }
   else 
   {
      if(enableModGUI)
      {
         ui.buttonMod_rest->setEnabled(true);
         ui.buttonMod_set->setEnabled(true);
         ui.buttonMod_mod->setEnabled(false);
      }
   }
   
   // ------- camwfs-align loop
   if(!m_camwfsAlignLoopWaiting)
   {
      ui.pupilAlign_loopSlider->setEnabled(true);
      if(m_camwfsAlignLoopState)
      {
         ui.pupilAlign_loopSlider->setSliderPosition(ui.pupilAlign_loopSlider->maximum());
      }
      else
      {
         ui.pupilAlign_loopSlider->setSliderPosition(ui.pupilAlign_loopSlider->minimum());
      }
   }
   
   // ------Pupil Fitting
   
   if( !(m_camwfsfitState == "READY" || m_camwfsfitState == "OPERATING"))
   {
      camwfsfitSetEnabled(false);
   }
   else
   {
      camwfsfitSetEnabled(true);

      double m1, m2, m3,m4;
      
      if(ui.setDelta->checkState() == Qt::Checked)
      {
         double ave = 0.25*(m_med1 + m_med2 + m_med3 + m_med4);
         m1 = m_med1-ave;
         m2 = m_med2-ave;
         m3 = m_med3-ave;
         m4 = m_med4-ave;
      }
      else
      {
         m1 = m_med1;
         m2 = m_med2;
         m3 = m_med3;
         m4 = m_med4;
      }
      
      snprintf(str, 16, "%0.1f", m1);
      ui.med1->setText(str);
      
      snprintf(str, 16, "%0.1f", m2);
      ui.med2->setText(str);
      
      snprintf(str, 16, "%0.1f", m3);
      ui.med3->setText(str);
      
      snprintf(str, 16, "%0.1f", m4);
      ui.med4->setText(str);
      
      double x1 = m_x1;
      double y1 = m_y1;
      double D1 = m_D1;
      double x2 = m_x2;
      double y2 = m_y2;
      double D2 = m_D2;
      double x3 = m_x3;
      double y3 = m_y3;
      double D3 = m_D3;
      double x4 = m_x4;
      double y4 = m_y4;
      double D4 = m_D4;
      
      if(ui.setDelta_pup->checkState() == Qt::Checked)
      {
         x1 -= m_setx1;
         y1 -= m_sety1;
         D1 -= m_setD1;
         
         x2 -= m_setx2;
         y2 -= m_sety2;
         D2 -= m_setD2;
         
         x3 -= m_setx3;
         y3 -= m_sety3;
         D3 -= m_setD3;
         
         x4 -= m_setx4;
         y4 -= m_sety4;
         D4 -= m_setD4;
      }
      
      snprintf(str, 16, "%0.2f", D1);
      ui.coordLL_D->setText(str);
     
      snprintf(str, 16, "%0.2f", D2);
      ui.coordLR_D->setText(str);
     
      snprintf(str, 16, "%0.2f", D3);         
      ui.coordUL_D->setText(str);

      snprintf(str, 16, "%0.2f", D4);
      ui.coordUR_D->setText(str);

      snprintf(str, 16, "%0.2f", x1);
      ui.coordLL_x->setText(str);
      
      snprintf(str, 16, "%0.2f", x2);
      ui.coordLR_x->setText(str);

      snprintf(str, 16, "%0.2f", x3);
      ui.coordUL_x->setText(str);
      
      snprintf(str, 16, "%0.2f", x4);
      ui.coordUR_x->setText(str);
            
      snprintf(str, 16, "%0.2f", y1);
      ui.coordLL_y->setText(str);
      
      snprintf(str, 16, "%0.2f", y2);
      ui.coordLR_y->setText(str);
      
      snprintf(str, 16, "%0.2f", y3);
      ui.coordUL_y->setText(str);
      
      
      snprintf(str, 16, "%0.2f", y4);
      ui.coordUR_y->setText(str);
      
      
      snprintf(str, 16, "%0.2f", 0.25*(D1+D2+D3+D4));
      ui.coordAvg_D->setText(str);
            
      snprintf(str, 16, "%0.2f", 0.25*(x1+x2+x3+x4));
      ui.coordAvg_x->setText(str);
      
      snprintf(str, 16, "%0.2f", 0.25*(y1+y2+y3+y4));
      ui.coordAvg_y->setText(str);
      
   }
   
   // ------ camwfs averaging 
   if( m_camwfsavgState == "READY" || m_camwfsavgState == "OPERATING")
   {
      ui.fitAvgTime->setEnabled(true);
   }
   else
   {
      ui.fitAvgTime->setEnabled(false);
   }
   
   // ------ dmtweeter
   
   if(m_dmtweeterState == "READY" || m_dmtweeterState == "OPERATING")
   {
      ui.buttonTweeterTest_set->setEnabled(true);
      if(m_dmtweeterTestSet)
      {
         ui.buttonTweeterTest_set->setText("zero test");
      }
      else
      {
         ui.buttonTweeterTest_set->setText("set test");
      }
   }
   else
   {
      ui.buttonTweeterTest_set->setEnabled(false);
      ui.buttonTweeterTest_set->setText("set test");
   }

   // ------ dmncpc
   
   if(m_dmncpcState == "READY" || m_dmncpcState == "OPERATING")
   {
      ui.buttonNCPCTest_set->setEnabled(true);
      if(m_dmncpcTestSet)
      {
         ui.buttonNCPCTest_set->setText("zero test");
      }
      else
      {
         ui.buttonNCPCTest_set->setText("set test");
      }
   }
   else
   {
      ui.buttonNCPCTest_set->setEnabled(false);
      ui.buttonNCPCTest_set->setText("set test");
   }

   // ------ Pupil Steering
   bool enablePupFSM = true;
   bool enablePupFSMArrows = true;
   
   if(m_pupFsmState == "READY")
   {
      ui.pupState->setEnabled(true);
      ui.pupCh1->setEnabled(true);
      ui.pupCh2->setEnabled(true);
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(true);
   }
   else if(m_pupFsmState == "NOTHOMED")
   {
      ui.pupState->setEnabled(true);
      ui.pupCh1->setEnabled(false);
      ui.pupCh2->setEnabled(false);
      ui.buttonPup_set->setEnabled(true);
      ui.buttonPup_rest->setEnabled(false);
      enablePupFSMArrows = false;
   }
   else if(m_pupFsmState == "HOMING")
   {
      ui.pupState->setEnabled(true);
      ui.pupCh1->setEnabled(false);
      ui.pupCh2->setEnabled(false);
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(true);
      enablePupFSMArrows = false;
   }
   else
   {
      enablePupFSM = false;
      if(m_pupFsmState == "")
      {
         ui.pupState->setEnabled(false);
      }
      else 
      {
         ui.pupState->setEnabled(true);
      }
   }

   if(enablePupFSM)
   {
      if(enablePupFSMArrows)
      {
         ui.button_pup_ul->setEnabled(true); 
         ui.button_pup_ur->setEnabled(true); 
         ui.button_pup_scale->setEnabled(true);
         ui.button_pup_dl->setEnabled(true);
         ui.button_pup_dr->setEnabled(true);
      }
      else
      {
         ui.button_pup_ul->setEnabled(false); 
         ui.button_pup_ur->setEnabled(false); 
         ui.button_pup_scale->setEnabled(false);
         ui.button_pup_dl->setEnabled(false);
         ui.button_pup_dr->setEnabled(false);
      }
   }
   else
   {

      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(false);
      ui.pupCh1->setEnabled(false);
      ui.pupCh2->setEnabled(false);
      
      ui.button_pup_ul->setEnabled(false); 
      ui.button_pup_ur->setEnabled(false); 
      ui.button_pup_scale->setEnabled(false);
      ui.button_pup_dl->setEnabled(false);
      ui.button_pup_dr->setEnabled(false);
   }
   
   // ------ TTM Peri
   bool enableTTMPeriFSM = true;
   bool enableTTMPeriFSMArrows = true;
   
   if(m_ttmPeriFsmState == "READY")
   {
      ui.ttmPeriState->setEnabled(true);
      ui.ttmPeriCh1->setEnabled(false);
      ui.ttmPeriCh2->setEnabled(false);
      ui.button_ttmPeri_set->setEnabled(false);
      ui.button_ttmPeri_rest->setEnabled(true);

      enableTTMPeriFSMArrows = false;
   }
   else if(m_ttmPeriFsmState == "OPERATING")
   {
      ui.ttmPeriState->setEnabled(true);
      ui.ttmPeriCh1->setEnabled(true);
      ui.ttmPeriCh2->setEnabled(true);
      ui.button_ttmPeri_set->setEnabled(false);
      ui.button_ttmPeri_rest->setEnabled(true);
      enableTTMPeriFSMArrows = true;
   }
   else
   {
      enableTTMPeriFSM = false;
      
      if(m_ttmPeriFsmState == "")
      {
         ui.ttmPeriState->setEnabled(false);
      }
      else 
      {
         ui.ttmPeriState->setEnabled(true);
      }

      ui.ttmPeriCh1->setEnabled(false);
      ui.ttmPeriCh2->setEnabled(false);
      ui.button_ttmPeri_set->setEnabled(false);
      ui.button_ttmPeri_rest->setEnabled(false);
   }

   if(enableTTMPeriFSM)
   {
      if(enableTTMPeriFSMArrows)
      {
         ui.button_ttmPeri_l->setEnabled(true); 
         ui.button_ttmPeri_r->setEnabled(true); 
         ui.button_ttmPeri_scale->setEnabled(true);
         ui.button_ttmPeri_u->setEnabled(true);
         ui.button_ttmPeri_d->setEnabled(true);
      }
      else
      {
         ui.button_ttmPeri_l->setEnabled(false); 
         ui.button_ttmPeri_r->setEnabled(false); 
         ui.button_ttmPeri_scale->setEnabled(false);
         ui.button_ttmPeri_u->setEnabled(false);
         ui.button_ttmPeri_d->setEnabled(false);
      }
   }
   else
   {  
      ui.button_ttmPeri_l->setEnabled(false); 
      ui.button_ttmPeri_r->setEnabled(false); 
      ui.button_ttmPeri_scale->setEnabled(false);
      ui.button_ttmPeri_u->setEnabled(false);
      ui.button_ttmPeri_d->setEnabled(false);
   }
   
   // --- camera lens
   
   
   if( (m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING") &&
          (m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING") )
   {
      
      camlensSetEnabled(true);
      
      if(m_camlensxFsmState == "OPERATING" || m_camlensyFsmState == "OPERATING")
      {
         ui.camlens_fsm->setText("OPERATING");
      }
      else
      {
         ui.camlens_fsm->setText("READY");
      }
   }
   else
   {
      camlensSetEnabled(false);
      
      if(m_camlensxFsmState == "HOMING" || m_camlensyFsmState == "HOMING")
      {
         ui.camlens_fsm->setEnabled(true);
         ui.camlens_fsm->setText("HOMING");
      }
      else if(m_camlensxFsmState == "POWERON" || m_camlensyFsmState == "POWERON")
      {
         ui.camlens_fsm->setEnabled(true);
         ui.camlens_fsm->setText("POWERON");
      }
      else if(m_camlensxFsmState == "POWEROFF" && m_camlensyFsmState == "POWEROFF")
      {
         ui.camlens_fsm->setEnabled(true);
         ui.camlens_fsm->setText("POWEROFF");
      }
      else
      {
         if(m_camlensxFsmState == "" && m_camlensyFsmState == "")
         {
            ui.camlens_fsm->setEnabled(false);
            ui.camlens_fsm->setText("STATE UNKNOWN");
         }
         else
         {
            ui.camlens_fsm->setEnabled(true);
            ui.camlens_fsm->setText(m_camlensxFsmState.c_str());
         }
      }
   }
   
   ui.pupilAlign_gain->updateGUI();

   ui.fitThreshold->updateGUI();
   ui.fitAvgTime->updateGUI();
   ui.camlens_x_pos->updateGUI();
   ui.camlens_y_pos->updateGUI();

} //updateGUI()

// ------------- modttm
   
void pupilGuide::on_buttonMod_mod_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = 4;
   
   sendNewProperty(ip);
}
   
void pupilGuide::on_buttonMod_set_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = 3;
   sendNewProperty(ip);
}
   
void pupilGuide::on_buttonMod_rest_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = 1;
   
   sendNewProperty(ip);
}
   
void pupilGuide::on_button_scalemodcamwfs_pressed()
{
   double scaletarg = 0.99990082103502819777 * m_camwfsFreq;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modFrequency");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = scaletarg;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_ttmtel_pressed()
{
   if(m_tipmovewhat == MOVE_TTM) 
   {
      m_tipmovewhat = MOVE_WOOF;
      ui.button_ttmtel->setText("move woofer");
   }
   else if(m_tipmovewhat == MOVE_WOOF) 
   {
      m_tipmovewhat = MOVE_TEL;
      ui.button_ttmtel->setText("move telescope");
   }
   else
   {
      m_tipmovewhat = MOVE_TTM;
      ui.button_ttmtel->setText("move ttm");
   }
}

void pupilGuide::on_button_tip_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = m_stepSize;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = 0;
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, 0, m_stepSize);
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
      
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = m_stepSize*5.;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = 0;
   }
   else return;

   sendNewProperty(ip);
   
}

void pupilGuide::on_button_tip_ul_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = m_stepSize/sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = -m_stepSize/sqrt(2.);
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, m_stepSize/sqrt(2.), m_stepSize/sqrt(2.));
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = m_stepSize*5./sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize*5./sqrt(2.);
   }

   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = 0;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = -m_stepSize;
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, m_stepSize, 0);
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = 0;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = -m_stepSize*5.;
   }

   sendNewProperty(ip);
   
}

void pupilGuide::on_button_tip_dl_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
  {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = -m_stepSize/sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = -m_stepSize/sqrt(2.);
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, m_stepSize/sqrt(2.), -m_stepSize/sqrt(2.));
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = -m_stepSize*5./sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = -m_stepSize*5./sqrt(2.);
   }

   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = -m_stepSize;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = 0;
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, 0, -m_stepSize);
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = -m_stepSize*5.;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = 0;
   }

   sendNewProperty(ip);
}

void pupilGuide::on_button_tip_dr_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = -m_stepSize/sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize/sqrt(2.);
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, -m_stepSize/sqrt(2.), -m_stepSize/sqrt(2.));
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = -m_stepSize*5./sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize*5./sqrt(2.);
   }
   else return;

   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = 0;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize;
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, -m_stepSize, 0);
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = 0;
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize*5.;
   }
   else return;

   sendNewProperty(ip);
   
}

void pupilGuide::on_button_tip_ur_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_TTM)
   {
      ip.setDevice("modwfs");
      ip.setName("offset");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = m_stepSize/sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize/sqrt(2.);
   }
   else if(m_tipmovewhat == MOVE_WOOF)
   {
      double tip, tilt;
      wooferTipTilt(tip, tilt, -m_stepSize/sqrt(2.), m_stepSize/sqrt(2.));
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0000"));
      ip.add(pcf::IndiElement("0001"));
      ip["0000"] = m_tip + tip;
      ip["0001"] = m_tilt + tilt;
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("y"));
      ip["y"] = m_stepSize*5./sqrt(2.);
      ip.add(pcf::IndiElement("x"));
      ip["x"] = m_stepSize*5./sqrt(2.);
   }
   else return;

   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_scale_pressed()
{
   if(((int) (100*m_stepSize)) == 100)
   {
      m_stepSize = 0.5;
   }
   else if(((int) (100*m_stepSize)) == 50)
   {
      m_stepSize = 0.1;
   }
   else if(((int) (100*m_stepSize)) == 10)
   {
      m_stepSize = 0.05;
   }
   else if(((int) (100*m_stepSize)) == 5)
   {
      m_stepSize = 0.01;
   }
   else if(((int) (100*m_stepSize)) == 1)
   {
      m_stepSize = 1.0;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.button_tip_scale->setText(ss);
   

}

void pupilGuide::on_button_focus_p_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   

   if(m_tipmovewhat == MOVE_WOOF)
   {
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0002"));
      ip["0002"] = m_focus + m_focusStepSize*0.2;
      
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("z"));
      ip["z"] = m_stepSize*5.;
   }
   else return;

   sendNewProperty(ip);
   
}

void pupilGuide::on_button_focus_m_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   if(m_tipmovewhat == MOVE_WOOF)
   {
      
      ip.setDevice("wooferModes");
      ip.setName("target_amps");
      ip.add(pcf::IndiElement("0002"));
      ip["0002"] = m_focus - m_focusStepSize*0.2;
      
   }
   else if(m_tipmovewhat == MOVE_TEL)
   {
      ip.setDevice("tcsi");
      ip.setName("pyrNudge");
      ip.add(pcf::IndiElement("z"));
      ip["z"] = -m_stepSize*5.;
   }
   else return;

   sendNewProperty(ip);
}

void pupilGuide::on_button_focus_scale_pressed()
{
   if(((int) (100*m_focusStepSize)) == 100)
   {
      m_focusStepSize = 0.5;
   }
   else if(((int) (100*m_focusStepSize)) == 50)
   {
      m_focusStepSize = 0.1;
   }
   else if(((int) (100*m_focusStepSize)) == 10)
   {
      m_focusStepSize = 0.05;
   }
   else if(((int) (100*m_focusStepSize)) == 5)
   {
      m_focusStepSize = 0.01;
   }
   else if(((int) (100*m_focusStepSize)) == 1)
   {
      m_focusStepSize = 1.0;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_focusStepSize);
   ui.button_focus_scale->setText(ss);
}

//----------- dmtweeter

void pupilGuide::on_buttonTweeterTest_set_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice("dmtweeter");
   ip.setName("test_set");
   ip.add(pcf::IndiElement("toggle"));

   if(m_dmtweeterTestSet)
   {
      ip["toggle"].setSwitchState(pcf::IndiElement::Off);
   }
   else
   {
      ip["toggle"].setSwitchState(pcf::IndiElement::On);
   }

   sendNewProperty(ip);
}

//----------- dmtweeter

void pupilGuide::on_buttonNCPCTest_set_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice("dmncpc");
   ip.setName("test_set");
   ip.add(pcf::IndiElement("toggle"));

   if(m_dmncpcTestSet)
   {
      ip["toggle"].setSwitchState(pcf::IndiElement::Off);
   }
   else
   {
      ip["toggle"].setSwitchState(pcf::IndiElement::On);
   }

   sendNewProperty(ip);
}

//----------- ttmpupil

void pupilGuide::on_buttonPup_rest_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice("ttmpupil");
   ip.setName("releaseDM");
   ip.add(pcf::IndiElement("request"));
   ip["request"].setSwitchState(pcf::IndiElement::On);
   
   sendNewProperty(ip);
   
}

void pupilGuide::on_buttonPup_set_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
   ip.setDevice("ttmpupil");
   ip.setName("initDM");
   ip.add(pcf::IndiElement("request"));
   ip["request"].setSwitchState(pcf::IndiElement::On);
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_camera_pressed()
{
   if(m_pupCam == FLOWFS)
   {
      m_pupCam = CAMSCIS;
      ui.button_camera->setText("camsci1/2");
   }
   else
   {
      m_pupCam = FLOWFS;
      ui.button_camera->setText("flowfs");
   }
}

void pupilGuide::on_button_pup_ul_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));

   if(m_pupCam == FLOWFS)
   {
      ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip);
   
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   
   if(m_pupCam == FLOWFS)
   {
      ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_dl_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   
   if(m_pupCam == FLOWFS)
   {
      ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   
   if(m_pupCam == FLOWFS)
   {
      ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_dr_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   
   if(m_pupCam == FLOWFS)
   {
      ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   
   if(m_pupCam == FLOWFS)
   {
      ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_ur_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   if(m_pupCam == FLOWFS)
   {
      ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   
   if(m_pupCam == FLOWFS)
   {
      ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   }
   else if(m_pupCam == LLOWFS)
   {

   }
   else if(m_pupCam == CAMSCIS)
   {
      ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   }

   sendNewProperty(ip2);
   
}

void pupilGuide::on_button_pup_scale_pressed()
{
   if(((int) (100*m_pupStepSize)) == 100)
   {
      m_pupStepSize = 0.5;
   }
   else if(((int) (100*m_pupStepSize)) == 50)
   {
      m_pupStepSize = 0.1;
   }
   else if(((int) (100*m_pupStepSize)) == 10)
   {
      m_pupStepSize = 0.05;
   }
   else if(((int) (100*m_pupStepSize)) == 5)
   {
      m_pupStepSize = 0.01;
   }
   else if(((int) (100*m_pupStepSize)) == 1)
   {
      m_pupStepSize = 1.0;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_pupStepSize);
   ui.button_pup_scale->setText(ss);
}

void pupilGuide::on_button_ttmPeri_rest_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
    ip.setDevice("ttmperi");
    ip.setName("set");
    ip.add(pcf::IndiElement("toggle"));
    ip["toggle"].setSwitchState(pcf::IndiElement::Off);
   
    sendNewProperty(ip);
}

void pupilGuide::on_button_ttmPeri_set_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   
    ip.setDevice("ttmperi");
    ip.setName("set");
    ip.add(pcf::IndiElement("toggle"));
    ip["toggle"].setSwitchState(pcf::IndiElement::On);
   
    sendNewProperty(ip);
}

void pupilGuide::on_button_ttmPeri_l_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
    ip.setDevice("ttmperi");
    ip.setName("axis1_voltage");
    ip.add(pcf::IndiElement("target"));

    if(m_pupCam == FLOWFS)
    {
        ip["target"] = m_ttmPeriCh1 - m_ttmPeriStepSize;
    }
    else if(m_pupCam == LLOWFS)
    {

    }
    else if(m_pupCam == CAMSCIS)
    {
        ip["target"] = m_ttmPeriCh1 - m_ttmPeriStepSize;
    }

   sendNewProperty(ip);

}

void pupilGuide::on_button_ttmPeri_r_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
    ip.setDevice("ttmperi");
    ip.setName("axis1_voltage");
    ip.add(pcf::IndiElement("target"));

    if(m_pupCam == FLOWFS)
    {
        ip["target"] = m_ttmPeriCh1 + m_ttmPeriStepSize;
    }
    else if(m_pupCam == LLOWFS)
    {

    }
    else if(m_pupCam == CAMSCIS)
    {
        ip["target"] = m_ttmPeriCh1 + m_ttmPeriStepSize;
    }

   sendNewProperty(ip);


}

void pupilGuide::on_button_ttmPeri_u_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
    ip.setDevice("ttmperi");
    ip.setName("axis2_voltage");
    ip.add(pcf::IndiElement("target"));

    if(m_pupCam == FLOWFS)
    {
        ip["target"] = m_ttmPeriCh2 + m_ttmPeriStepSize;
    }
    else if(m_pupCam == LLOWFS)
    {

    }
    else if(m_pupCam == CAMSCIS)
    {
        ip["target"] = m_ttmPeriCh2 + m_ttmPeriStepSize;
    }

    sendNewProperty(ip);

}

void pupilGuide::on_button_ttmPeri_d_pressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
    ip.setDevice("ttmperi");
    ip.setName("axis2_voltage");
    ip.add(pcf::IndiElement("target"));

    if(m_pupCam == FLOWFS)
    {
        ip["target"] = m_ttmPeriCh2 - m_ttmPeriStepSize;
    }
    else if(m_pupCam == LLOWFS)
    {

    }
    else if(m_pupCam == CAMSCIS)
    {
        ip["target"] = m_ttmPeriCh2 - m_ttmPeriStepSize;
    }

    sendNewProperty(ip);
}

void pupilGuide::on_button_ttmPeri_scale_pressed()
{
   if(((int) (m_ttmPeriStepSize)) == 50)
   {
      m_ttmPeriStepSize = 25;
   }
   else if(((int) (m_ttmPeriStepSize)) == 25)
   {
      m_ttmPeriStepSize = 10;
   }
   else if(((int) (m_ttmPeriStepSize)) == 10)
   {
      m_ttmPeriStepSize = 1;
   }
   else
   {
      m_ttmPeriStepSize = 50;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_ttmPeriStepSize/100.);
   ui.button_ttmPeri_scale->setText(ss);
}


void pupilGuide::on_pupilAlign_loopSlider_sliderReleased()
{
   double relpos = ((double)(ui.pupilAlign_loopSlider->sliderPosition() - ui.pupilAlign_loopSlider->minimum()))/(ui.pupilAlign_loopSlider->maximum() - ui.pupilAlign_loopSlider->minimum());
   
   if(m_camwfsAlignLoopState)
   {
      if(relpos > 0.1)
      {
         ui.pupilAlign_loopSlider->setSliderPosition(ui.pupilAlign_loopSlider->maximum());
         ui.pupilAlign_loopSlider->setEnabled(true);
         m_camwfsAlignLoopWaiting = false;
         return;
      }
   }
   else 
   {
      if(relpos < 0.9)
      {
         ui.pupilAlign_loopSlider->setSliderPosition(ui.pupilAlign_loopSlider->minimum());
         ui.pupilAlign_loopSlider->setEnabled(true);
         m_camwfsAlignLoopWaiting = false;
         return;
      }
   }

   ui.pupilAlign_loopSlider->setEnabled(false);
   m_camwfsAlignLoopWaiting = true;
   m_camwfsAlignLoopWaitTimer->start(2000);
   
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice("camwfs-align");
   ipFreq.setName("loop_state");
   ipFreq.add(pcf::IndiElement("toggle"));
   
   if(relpos >= 0.9)
   {
      ipFreq["toggle"] = pcf::IndiElement::On;
   }
   else
   {
      ipFreq["toggle"] = pcf::IndiElement::Off;
   }
   
   sendNewProperty(ipFreq);
}

void pupilGuide::on_pupilAlign_loopSlider_sliderPressed()
{
   m_camwfsAlignLoopWaiting = true;
   m_camwfsAlignLoopWaitTimer->start(2000);
}

void pupilGuide::on_pupilAlign_loopSlider_sliderMoved(int p)
{
   static_cast<void>(p);

   m_camwfsAlignLoopWaiting = true;
   m_camwfsAlignLoopWaitTimer->start(2000);
}

void pupilGuide::camwfsAlignLoopWaitTimerOut()
{
   m_camwfsAlignLoopWaiting = false;
}

void pupilGuide::toggleExpFit(bool st)
{

   ui.labelD->setVisible(st);

   ui.labelUR->setVisible(st);
   ui.coordUR_x->setVisible(st);
   ui.coordUR_y->setVisible(st);
   ui.coordUR_D->setVisible(st);

   ui.labelUL->setVisible(st);
   ui.coordUL_x->setVisible(st);
   ui.coordUL_y->setVisible(st);
   ui.coordUL_D->setVisible(st);

   ui.labelLR->setVisible(st);
   ui.coordLR_x->setVisible(st);
   ui.coordLR_y->setVisible(st);
   ui.coordLR_D->setVisible(st);

   ui.labelLL->setVisible(st);
   ui.coordLL_x->setVisible(st);
   ui.coordLL_y->setVisible(st);
   ui.coordLL_D->setVisible(st);

   ui.coordAvg_D->setVisible(st);

   if(st)
   {
      ui.buttonExpFit->setText("--");
   }
   else
   {
      ui.buttonExpFit->setText("|");
   }
}


void pupilGuide::on_buttonExpFit_pressed()
{
   bool st = !ui.labelD->isVisible();
   toggleExpFit(st);
}


void pupilGuide::on_button_camlens_u_pressed()
{
   if(m_camlensyFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensy");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensy_pos - m_camlensStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_camlens_l_pressed()
{
   if(m_camlensxFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensx");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensx_pos - m_camlensStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_camlens_d_pressed()
{
   if(m_camlensyFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensy");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensy_pos + m_camlensStepSize;
   
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_camlens_r_pressed()
{
   if(m_camlensxFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensx");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensx_pos + m_camlensStepSize;
   sendNewProperty(ip);
}

void pupilGuide::on_button_camlens_scale_pressed()
{
   if(((int) (1000*m_camlensStepSize+0.5)) == 5)
   {
      m_camlensStepSize = 0.05;
   }
   else if(((int) (1000*m_camlensStepSize+0.5)) == 50)
   {
      m_camlensStepSize = 0.025;
   }
   else if(((int) (1000*m_camlensStepSize+0.5)) == 25)
   {
      m_camlensStepSize = 0.01;
   }
   else if(((int) (1000*m_camlensStepSize+0.5)) == 10)
   {
      m_camlensStepSize = 0.005;
   }
/*   else if(((int) (1000*m_camlensStepSize+0.5)) == 5)
   {
      m_camlensStepSize = 0.001;
   }*/
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_camlensStepSize*10);
   ui.button_camlens_scale->setText(ss);


}


} //namespace xqt
   
#include "moc_pupilGuide.cpp"

#endif
