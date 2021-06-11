#ifndef pupilGuide_hpp
#define pupilGuide_hpp

#include <cmath>
#include <unistd.h>

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_pupilGuide.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class pupilGuide : public QDialog, public multiIndiSubscriber
{
   Q_OBJECT
   
protected:
   
   std::string m_appState;
   
   QMutex m_mutex;
   
   std::string m_modFsmState;
   int m_modState {0};
   
   double m_modCh1 {0};
   double m_modCh2 {0};
   
   double m_modFreq {0};
   double m_modFreq_tgt{0};
   
   double m_modRad {0};
   double m_modRad_tgt{0};
   


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

   
   
   std::string m_pupFsmState;
   double m_pupCh1 {0};
   double m_pupCh2 {0};
   
   
   
   float m_stepSize {0.1};
   
   
   float m_pupStepSize {0.1};
    
   std::string m_camlensxFsmState;
   std::string m_camlensyFsmState;
   float m_camlensx_pos {0};
   float m_camlensy_pos {0};
   
   float m_camlensStepSize {0.01};
   
public:
   pupilGuide( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   ~pupilGuide();
   
   int subscribe( multiIndiPublisher * publisher );
                               
   virtual void onConnect();
   virtual void onDisconnect();
   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
   void modGUISetEnable( bool enableModGUI,
                         bool enableModArrows
                       );
   
   void camwfsfitSetEnabled(bool enabled);
   
   void camlensSetEnabled(bool enabled);
   
public slots:
   void updateGUI();
   
   void on_button_tip_u_pressed();
   void on_button_tip_ul_pressed();
   void on_button_tip_l_pressed();
   void on_button_tip_dl_pressed();
   void on_button_tip_d_pressed();
   void on_button_tip_dr_pressed();
   void on_button_tip_r_pressed();
   void on_button_tip_ur_pressed();
   void on_button_tip_scale_pressed();
   
   
   void on_button_pup_u_pressed();
   void on_button_pup_ul_pressed();
   void on_button_pup_l_pressed();
   void on_button_pup_dl_pressed();
   void on_button_pup_d_pressed();
   void on_button_pup_dr_pressed();
   void on_button_pup_r_pressed();
   void on_button_pup_ur_pressed();
   void on_button_pup_scale_pressed();
   
   void on_modFreq_target_returnPressed();
   void on_modRad_target_returnPressed();
   void on_buttonMod_mod_pressed();
   void on_buttonMod_set_pressed();
   void on_buttonMod_rest_pressed();
   
   void on_buttonPup_rest_pressed();
   void on_buttonPup_set_pressed();
   
   void on_fitThreshold_target_returnPressed();
   void on_nAverage_target_returnPressed();
   
   void on_button_camlens_u_pressed();
   void on_button_camlens_ul_pressed();
   void on_button_camlens_l_pressed();
   void on_button_camlens_dl_pressed();
   void on_button_camlens_d_pressed();
   void on_button_camlens_dr_pressed();
   void on_button_camlens_r_pressed();
   void on_button_camlens_ur_pressed();
   void on_button_camlens_scale_pressed();
private:
     
   Ui::pupilGuide ui;
};

pupilGuide::pupilGuide( QWidget * Parent, Qt::WindowFlags f) : QDialog(Parent, f)
{
   ui.setupUi(this);
   ui.modState->setProperty("isStatus", true);
   ui.pupState->setProperty("isStatus", true);
   ui.camlens_fsm->setProperty("isStatus", true);
   ui.button_camlens_scale->setProperty("isScaleButton", true);
   ui.button_tip_scale->setProperty("isScaleButton", true);
   ui.button_pup_scale->setProperty("isScaleButton", true);
   
   QTimer *timer = new QTimer(this);
   connect(timer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   timer->start(250);
      
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.button_tip_scale->setText(ss);
   
   snprintf(ss, 5, "%0.2f", m_pupStepSize);
   ui.button_pup_scale->setText(ss);
   
   snprintf(ss, 5, "%0.2f", m_camlensStepSize);
   ui.button_camlens_scale->setText(ss);
   
}
   
pupilGuide::~pupilGuide()
{
}

int pupilGuide::subscribe( multiIndiPublisher * publisher )
{
   publisher->subscribeProperty(this, "modwfs", "modFrequency");
   publisher->subscribeProperty(this, "modwfs", "modRadius");
   publisher->subscribeProperty(this, "modwfs", "modState");
   publisher->subscribeProperty(this, "modwfs", "fsm");

   publisher->subscribeProperty(this, "camwfs-fit", "fsm");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant1");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant2");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant3");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant4");
   publisher->subscribeProperty(this, "camwfs-fit", "threshold");
   
   publisher->subscribeProperty(this, "camwfs-avg", "fsm");
   publisher->subscribeProperty(this, "camwfs-avg", "nAverage");

   publisher->subscribeProperty(this, "fxngenmodwfs", "C1ofst");
   publisher->subscribeProperty(this, "fxngenmodwfs", "C2ofst");
   
   publisher->subscribeProperty(this, "ttmpupil", "fsm");
   publisher->subscribeProperty(this, "ttmpupil", "pos_1");
   publisher->subscribeProperty(this, "ttmpupil", "pos_2");
   
   publisher->subscribeProperty(this, "stagecamlensx", "fsm");
   publisher->subscribeProperty(this, "stagecamlensx", "position");
   
   publisher->subscribeProperty(this, "stagecamlensy", "fsm");
   publisher->subscribeProperty(this, "stagecamlensy", "position");
   
   return 0;
}
   
void pupilGuide::onConnect()
{
   ui.labelModAndCenter->setEnabled(true);
   ui.labelPupilFitting->setEnabled(true);
   ui.labelPupilSteering->setEnabled(true);
   ui.camlens_label->setEnabled(true);
   
   setWindowTitle("Pupil Alignment");
}

void pupilGuide::onDisconnect()
{
   m_modFsmState = "";
   m_pupFsmState = "";
   m_camlensxFsmState = "";
   m_camlensyFsmState = "";
   m_camwfsavgState = "";
   m_camwfsfitState = "";

   ui.labelModAndCenter->setEnabled(false);
   ui.labelPupilFitting->setEnabled(false);
   ui.labelPupilSteering->setEnabled(false);
   ui.camlens_label->setEnabled(false);
   
   setWindowTitle("Pupil Alignment (disconnected)");
   
}
   
int pupilGuide::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   std::string dev = ipRecv.getDevice();
   if( dev == "camwfs-avg" || 
       dev == "camwfs-fit" || 
       dev == "modwfs" || 
       dev == "fxngenmodwfs" || 
       dev == "ttmpupil" || 
       dev == "stagecamlensx" || 
       dev == "stagecamlensy") 
   {
      return handleSetProperty(ipRecv);
   }
   
   return 0;
}

int pupilGuide::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::string dev = ipRecv.getDevice();
   
   //m_mutex.lock();
   if(dev == "camwfs-avg")
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
   else if(dev == "modwfs")
   {
      if(ipRecv.getName() == "modFrequency")
      {
         if(ipRecv.find("current"))
         {
            m_modFreq = ipRecv["current"].get<double>();
         }
         if(ipRecv.find("requested"))
         {
            m_modFreq_tgt = ipRecv["requested"].get<double>();
         }
      }
      
      if(ipRecv.getName() == "modRadius")
      {
         if(ipRecv.find("current"))
         {
            m_modRad = ipRecv["current"].get<double>();
         }
         if(ipRecv.find("requested"))
         {
            m_modRad_tgt = ipRecv["requested"].get<double>();
         }
      }
      
      if(ipRecv.getName() == "modState")
      {
         if(ipRecv.find("current"))
         {
            m_modState = ipRecv["current"].get<int>();
         }
      }
      
      if(ipRecv.getName() == "fsm")
      {
         if(ipRecv.find("state"))
         {
            m_modFsmState = ipRecv["state"].get<std::string>();
         }
      }
   }
   else if(dev == "fxngenmodwfs")
   {
      if(ipRecv.getName() == "C1ofst")
      {
         if(ipRecv.find("value"))
         {
            m_modCh1 = ipRecv["value"].get<double>();
         }
      }
      if(ipRecv.getName() == "C2ofst")
      {
         if(ipRecv.find("value"))
         {
            m_modCh2 = ipRecv["value"].get<double>();
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
      
      if(ipRecv.getName() == "pos_1")
      {
         if(ipRecv.find("current"))
         {
            m_pupCh1 = ipRecv["current"].get<double>();
         }
      }
      
      if(ipRecv.getName() == "pos_2")
      {
         if(ipRecv.find("current"))
         {
            m_pupCh2 = ipRecv["current"].get<double>();
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
   
   return 0;
   
}

void pupilGuide::modGUISetEnable( bool enableModGUI,
                                  bool enableModArrows
                                )
{
   if(enableModGUI)
   {
      ui.modState->setEnabled(true);
      ui.labelFreqCurrent->setEnabled(true);
      ui.labelFreqTarget->setEnabled(true);
      ui.labelFreq->setEnabled(true);
      ui.labelRadius->setEnabled(true);
      ui.modFreq_current->setEnabled(true);
      ui.modRad_current->setEnabled(true);
      //ui.modFreq_target->setEnabled(true);
      //ui.modRad_target->setEnabled(true);
      //ui.buttonMod_rest->setEnabled(true);
      //ui.buttonMod_set->setEnabled(true);
      //ui.buttonMod_mod->setEnabled(true);
      ui.labelCh1->setEnabled(true);
      ui.modCh1->setEnabled(true);
      ui.labelCh2->setEnabled(true);
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
      }
   }
   else
   {
      ui.labelFreqCurrent->setEnabled(false);
      ui.labelFreqTarget->setEnabled(false);
      ui.labelFreq->setEnabled(false);
      ui.labelRadius->setEnabled(false);
      ui.modFreq_current->setEnabled(false);
      ui.modRad_current->setEnabled(false);
      ui.modFreq_target->setEnabled(false);
      ui.modRad_target->setEnabled(false);
      ui.buttonMod_rest->setEnabled(false);
      ui.buttonMod_set->setEnabled(false);
      ui.buttonMod_mod->setEnabled(false);
      ui.labelCh1->setEnabled(false);
      ui.modCh1->setEnabled(false);
      ui.labelCh2->setEnabled(false);
      ui.modCh2->setEnabled(false);
      
      ui.button_tip_ul->setEnabled(false); 
      ui.button_tip_u->setEnabled(false);
      ui.button_tip_ur->setEnabled(false); 
      ui.button_tip_l->setEnabled(false); 
      ui.button_tip_scale->setEnabled(false);
      ui.button_tip_r->setEnabled(false);
      ui.button_tip_dl->setEnabled(false);
      ui.button_tip_d->setEnabled(false);
      ui.button_tip_dr->setEnabled(false);
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
   ui.labelThreshCurrent->setEnabled(enabled);
   ui.labelThreshTarget->setEnabled(enabled);
   ui.labelThreshold->setEnabled(enabled);
   ui.fitThreshold_current->setEnabled(enabled);
   ui.fitThreshold_target->setEnabled(enabled);
   
   if(enabled == false)
   {
      ui.med1->setText("");
      ui.med2->setText("");
      ui.med3->setText("");
      ui.med4->setText("");
      ui.fitThreshold_current->setText("");
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
   ui.camlens_x_label->setEnabled(enabled);
   ui.camlens_x_pos->setEnabled(enabled);
   ui.camlens_y_label->setEnabled(enabled);
   ui.camlens_y_pos->setEnabled(enabled);
   ui.button_camlens_ul->setEnabled(enabled);
   ui.button_camlens_u->setEnabled(enabled);
   ui.button_camlens_ur->setEnabled(enabled);
   ui.button_camlens_l->setEnabled(enabled);
   ui.button_camlens_r->setEnabled(enabled);
   ui.button_camlens_dl->setEnabled(enabled);
   ui.button_camlens_d->setEnabled(enabled);
   ui.button_camlens_dr->setEnabled(enabled);
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
         enableModArrows = false;
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
   
   snprintf(str,sizeof(str), "%0.2f", m_modCh1);
   ui.modCh1->setText(str);
   snprintf(str,sizeof(str), "%0.2f", m_modCh2);
   ui.modCh2->setText(str);
   
   if(m_modState == 3 && enableModGUI)
   {
      ui.modFreq_current->setText("---");
      ui.modRad_current->setText("---");
      
      if(!ui.modFreq_target->hasFocus())
      {
         if(m_modFreq_tgt == 0)
         {
            ui.modFreq_target->setText("");
         }
         else
         {
            snprintf(str, sizeof(str),"%0.1f", m_modFreq_tgt);
            ui.modFreq_target->setText(str);
         }
      }
      
      if(!ui.modRad_target->hasFocus())
      {
         if(m_modRad_tgt == 0)
         {
            ui.modRad_target->setText("");
         }
         else
         {
            snprintf(str, sizeof(str),"%0.1f", m_modRad_tgt);
            ui.modRad_target->setText(str);
         }
      }
      
      ui.modFreq_target->setEnabled(true);
      ui.modRad_target->setEnabled(true);
      ui.buttonMod_rest->setEnabled(true);
      ui.buttonMod_set->setEnabled(false);
      ui.buttonMod_mod->setEnabled(true);
   }
   else if(m_modState == 4 && enableModGUI)
   {
      snprintf(str, sizeof(str),"%0.1f Hz", m_modFreq);
      ui.modFreq_current->setText(str);
      snprintf(str, sizeof(str),"%0.1f l/D", m_modRad);
      ui.modRad_current->setText(str);
      
      if(!ui.modFreq_target->hasFocus())
      {
         if(m_modFreq_tgt == 0)
         {
            ui.modFreq_target->setText("");
         }
         else
         {
            snprintf(str, sizeof(str),"%0.1f", m_modFreq_tgt);
            ui.modFreq_target->setText(str);
         }
      }
      
      if(!ui.modRad_target->hasFocus())
      {
         if(m_modRad_tgt == 0)
         {
            ui.modRad_target->setText("");
         }
         else
         {
            snprintf(str, sizeof(str),"%0.1f", m_modRad_tgt);
            ui.modRad_target->setText(str);
         }
      }
      
      ui.modFreq_target->setEnabled(true);
      ui.modRad_target->setEnabled(true);
      ui.buttonMod_rest->setEnabled(true);
      ui.buttonMod_set->setEnabled(true);
      ui.buttonMod_mod->setEnabled(true);
   }
   else 
   {
      ui.modFreq_current->setText("---");
      ui.modRad_current->setText("---");
      
      ui.modFreq_target->setText("");
      ui.modRad_target->setText("");
      
      if(enableModGUI)
      {
         ui.modFreq_target->setEnabled(false);
         ui.modRad_target->setEnabled(false);
         ui.buttonMod_rest->setEnabled(true);
         ui.buttonMod_set->setEnabled(true);
         ui.buttonMod_mod->setEnabled(false);
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
      
      snprintf(str, 16, "%0.4f", m_threshold_current);
      ui.fitThreshold_current->setText(str);
      
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
   if( !(m_camwfsavgState == "READY" || m_camwfsavgState == "OPERATING"))
   {
      //ui.labelThreshCurrent->setEnabled(false); //these may or may not be enabled by camwfs-fit above
      //ui.labelThreshTarget->setEnabled(false);
      ui.labelNoAvg->setEnabled(false);
      ui.nAverage_current->setEnabled(false);
      ui.nAverage_target->setEnabled(false);
      
      ui.nAverage_current->setText("");
   }
   else
   {
      ui.labelThreshCurrent->setEnabled(true); //always enable if we need them here
      ui.labelThreshTarget->setEnabled(true);
      ui.labelNoAvg->setEnabled(true);
      ui.nAverage_current->setEnabled(true);
      ui.nAverage_target->setEnabled(true);
      snprintf(str, 16, "%u", m_nAverage_current);
      ui.nAverage_current->setText(str);
   
   }
   
   // ------ Pupil Steering
   bool enablePupFSM = true;
   bool enablePupFSMArrows = true;
   
   if(m_pupFsmState == "READY")
   {
      ui.pupState->setText("SET");
      ui.pupState->setEnabled(true);
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(true);
   }
   else if(m_pupFsmState == "NOTHOMED")
   {
      ui.pupState->setText("RIP");
      ui.pupState->setEnabled(true);
      ui.buttonPup_set->setEnabled(true);
      ui.buttonPup_rest->setEnabled(false);
      enablePupFSMArrows = false;
   }
   else if(m_pupFsmState == "HOMING")
   {
      ui.pupState->setText("SETTING");
      ui.pupState->setEnabled(true);
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(true);
      enablePupFSMArrows = false;
   }
   else
   {
      enablePupFSM = false;
      if(m_pupFsmState == "")
      {
         ui.pupState->setText("STATE UNKNOWN");
         ui.pupState->setEnabled(false);
      }
      else 
      {
         ui.pupState->setEnabled(true);
         ui.pupState->setText(m_pupFsmState.c_str());
      }
   }

   if(enablePupFSM)
   {
      ui.buttonPup_set->setEnabled(true);
      ui.buttonPup_rest->setEnabled(true);
      ui.labelPupCh1->setEnabled(true); 
      ui.pupCh1->setEnabled(true);
      ui.labelPupCh2->setEnabled(true);
      ui.pupCh2->setEnabled(true);
      
      snprintf(str,sizeof(str), "%0.2f", m_pupCh1);
      ui.pupCh1->setText(str);
      snprintf(str,sizeof(str), "%0.2f", m_pupCh2);
      ui.pupCh2->setText(str);
      
      if(enablePupFSMArrows)
      {
         ui.button_pup_ul->setEnabled(true); 
         ui.button_pup_u->setEnabled(true);
         ui.button_pup_ur->setEnabled(true); 
         ui.button_pup_l->setEnabled(true); 
         ui.button_pup_scale->setEnabled(true);
         ui.button_pup_r->setEnabled(true);
         ui.button_pup_dl->setEnabled(true);
         ui.button_pup_d->setEnabled(true);
         ui.button_pup_dr->setEnabled(true);
      }
      else
      {
         ui.button_pup_ul->setEnabled(false); 
         ui.button_pup_u->setEnabled(false);
         ui.button_pup_ur->setEnabled(false); 
         ui.button_pup_l->setEnabled(false); 
         ui.button_pup_scale->setEnabled(false);
         ui.button_pup_r->setEnabled(false);
         ui.button_pup_dl->setEnabled(false);
         ui.button_pup_d->setEnabled(false);
         ui.button_pup_dr->setEnabled(false);
      }
   }
   else
   {
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(false);
      ui.labelPupCh1->setEnabled(false); 
      ui.pupCh1->setEnabled(false);
      ui.labelPupCh2->setEnabled(false);
      ui.pupCh2->setEnabled(false);
      
      ui.pupCh1->setText("");
      ui.pupCh2->setText("");
      
      ui.button_pup_ul->setEnabled(false); 
      ui.button_pup_u->setEnabled(false);
      ui.button_pup_ur->setEnabled(false); 
      ui.button_pup_l->setEnabled(false); 
      ui.button_pup_scale->setEnabled(false);
      ui.button_pup_r->setEnabled(false);
      ui.button_pup_dl->setEnabled(false);
      ui.button_pup_d->setEnabled(false);
      ui.button_pup_dr->setEnabled(false);
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
      
      snprintf(str, 16, "%0.4f", m_camlensx_pos);
      ui.camlens_x_pos->setText(str);
      snprintf(str, 16, "%0.4f", m_camlensy_pos);
      ui.camlens_y_pos->setText(str);
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
      
      ui.camlens_x_pos->setText("---");
      ui.camlens_y_pos->setText("---");
   }
      
} //updateGUI()

void pupilGuide::on_button_tip_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   
}

void pupilGuide::on_button_tip_ul_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize/sqrt(2.);
   ip.add(pcf::IndiElement("x"));
   ip["x"] = -m_stepSize/sqrt(2.);
   
   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = 0;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = -m_stepSize;
   
   sendNewProperty(ip);
   
}

void pupilGuide::on_button_tip_dl_pressed()
{
   
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = -m_stepSize/sqrt(2.);
   ip.add(pcf::IndiElement("x"));
   ip["x"] = -m_stepSize/sqrt(2.);
   
   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = -m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_tip_dr_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = -m_stepSize/sqrt(2.);
   ip.add(pcf::IndiElement("x"));
   ip["x"] = m_stepSize/sqrt(2.);
   
   sendNewProperty(ip);
   
   
}

void pupilGuide::on_button_tip_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = 0;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = m_stepSize;
   
   sendNewProperty(ip);
   
}

void pupilGuide::on_button_tip_ur_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize/sqrt(2.);
   ip.add(pcf::IndiElement("x"));
   ip["x"] = m_stepSize/sqrt(2.);
   
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


void pupilGuide::on_button_pup_l_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh1 + m_pupStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_pup_ul_pressed()
{
  pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip);
   
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_d_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_2");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh2 - m_pupStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_pup_dl_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_r_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh1 - m_pupStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_pup_dr_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip);
   
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_u_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_2");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh2 + m_pupStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_pup_ur_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("pos_1");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   
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

void pupilGuide::on_button_camlens_ul_pressed()
{
   if(m_camlensyFsmState != "READY" || m_camlensxFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensy");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensy_pos - sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("stagecamlensx");
   ip2.setName("position");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_camlensx_pos - sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip2);
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

void pupilGuide::on_button_camlens_dl_pressed()
{
   if(m_camlensyFsmState != "READY" || m_camlensxFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensy");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensy_pos + sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("stagecamlensx");
   ip2.setName("position");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_camlensx_pos - sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip2);
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

void pupilGuide::on_button_camlens_dr_pressed()
{
   if(m_camlensyFsmState != "READY" || m_camlensxFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensy");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensy_pos + sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("stagecamlensx");
   ip2.setName("position");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_camlensx_pos + sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip2);
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

void pupilGuide::on_button_camlens_ur_pressed()
{
   if(m_camlensyFsmState != "READY" || m_camlensxFsmState != "READY") return;
   
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("stagecamlensy");
   ip.setName("position");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = m_camlensy_pos - sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("stagecamlensx");
   ip2.setName("position");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_camlensx_pos + sqrt(2)*m_camlensStepSize;
   
   sendNewProperty(ip2);
}
void pupilGuide::on_button_camlens_scale_pressed()
{
   /*if(((int) (100*m_camlensStepSize)) == 100)
   {
      m_camlensStepSize = 0.5;
   }
   else if(((int) (100*m_camlensStepSize)) == 50)
   {
      m_camlensStepSize = 0.1;
   }
   else*/ 
   if(((int) (100*m_camlensStepSize)) == 10)
   {
      m_camlensStepSize = 0.05;
   }
   else if(((int) (100*m_camlensStepSize)) == 5)
   {
      m_camlensStepSize = 0.01;
   }
   else if(((int) (100*m_camlensStepSize)) == 1)
   {
      m_camlensStepSize = 0.1;
   }
   
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_camlensStepSize);
   ui.button_camlens_scale->setText(ss);


}

void pupilGuide::on_modFreq_target_returnPressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modFrequency");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = ui.modFreq_target->text().toDouble();
   
   sendNewProperty(ip);
   
}
   
void pupilGuide::on_modRad_target_returnPressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modRadius");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = ui.modRad_target->text().toDouble();
   
   sendNewProperty(ip);
}
   
void pupilGuide::on_buttonMod_mod_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = 4;
   
   sendNewProperty(ip);
}
   
void pupilGuide::on_buttonMod_set_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = 3;
   sendNewProperty(ip);
}
   
void pupilGuide::on_buttonMod_rest_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("modwfs");
   ip.setName("modState");
   ip.add(pcf::IndiElement("requested"));
   ip["requested"] = 1;
   
   sendNewProperty(ip);
}
   
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

void pupilGuide::on_fitThreshold_target_returnPressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("camwfs-fit");
   ip.setName("threshold");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.fitThreshold_target->text().toDouble();
   
   sendNewProperty(ip);
}

void pupilGuide::on_nAverage_target_returnPressed()
{
    pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("camwfs-avg");
   ip.setName("nAverage");
   ip.add(pcf::IndiElement("target"));
   ip["target"] = ui.nAverage_target->text().toDouble();
   
   sendNewProperty(ip);
}

} //namespace xqt
   
#include "moc_pupilGuide.cpp"

#endif
