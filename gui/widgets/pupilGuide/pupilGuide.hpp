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
   
   
   std::string m_pupFsmState;
   double m_pupCh1 {0};
   double m_pupCh2 {0};
   
   double m_threshold_current {0};
   double m_threshold_target {0};
   
   unsigned m_nAverage_current {0};
   unsigned m_nAverage_target {0};
   
   
   
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
                                   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
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
   void on_buttonScale_pressed();
   
   
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
//    QPalette p = palette();
//    synchActiveInactive(&p);
//    setPalette(p);
   
   ui.setupUi(this);
   //QPalette p = ui.buttonRest->palette();
   //synchActiveInactive(&p);
   //ui.buttonRest->setPalette(p);
   
   QTimer *timer = new QTimer(this);
   connect(timer, SIGNAL(timeout()), this, SLOT(updateGUI()));
   timer->start(250);
      
   char ss[5];
   snprintf(ss, 5, "%0.2f", m_stepSize);
   ui.buttonScale->setText(ss);
   
   snprintf(ss, 5, "%0.2f", m_pupStepSize);
   ui.button_pup_scale->setText(ss);
   
   snprintf(ss, 5, "%0.2f", m_camlensStepSize);
   ui.button_camlens_scale->setText(ss);
   
   ui.labelModAndCenter->setStyleSheet("*{color: white;}");
   ui.labelFreqCurrent->setStyleSheet("*{color: white;}");
   ui.labelFreqTarget->setStyleSheet("*{color: white;}");
   ui.labelFreq->setStyleSheet("*{color: white;}");
   ui.labelRadius->setStyleSheet("*{color: white;}");
   ui.modFreq_current->setStyleSheet("*{color: white;}");
   ui.modFreq_target->setStyleSheet("*{color: white;}");
   ui.modRad_current->setStyleSheet("*{color: white;}");
   ui.modRad_target->setStyleSheet("*{color: white;}");
   ui.buttonMod_rest->setStyleSheet("*{color: white;}");
   ui.buttonMod_set->setStyleSheet("*{color: white;}");
   ui.buttonMod_mod->setStyleSheet("*{color: white;}");
   
   ui.labelCh1->setStyleSheet("*{color: white;}");
   ui.modCh1->setStyleSheet("*{color: white;}");
   ui.labelCh2->setStyleSheet("*{color: white;}");
   ui.modCh2->setStyleSheet("*{color: white;}");
   
   ui.labelMedianFluxes->setStyleSheet("*{color: white;}");
   ui.med1->setStyleSheet("*{color: white;}");
   ui.med2->setStyleSheet("*{color: white;}");
   ui.med3->setStyleSheet("*{color: white;}");
   ui.med4->setStyleSheet("*{color: white;}");
   
   ui.setDelta->setStyleSheet("*{color: white;}");
   
   ui.buttonScale->setStyleSheet("*{color: white;}");
   
   ui.labelPupilFitting->setStyleSheet("*{color: white;}");
   ui.labelThreshCurrent->setStyleSheet("*{color: white;}");
   ui.labelThreshTarget->setStyleSheet("*{color: white;}");
   ui.labelThreshold->setStyleSheet("*{color: white;}");
   ui.labelNoAvg->setStyleSheet("*{color: white;}");
   ui.fitThreshold_current->setStyleSheet("*{color: white;}");
   ui.fitThreshold_target->setStyleSheet("*{color: white;}");
   ui.nAverage_current->setStyleSheet("*{color: white;}");
   ui.nAverage_target->setStyleSheet("*{color: white;}");
   
   ui.camlens_label->setStyleSheet("*{color: white;}");
   ui.camlens_x_label->setStyleSheet("*{color: white;}");
   ui.camlens_x_pos->setStyleSheet("*{color: white;}");
   ui.camlens_y_label->setStyleSheet("*{color: white;}");
   ui.camlens_y_pos->setStyleSheet("*{color: white;}");
   ui.button_camlens_scale->setStyleSheet("*{color: white;}");
   
   ui.labelPupilSteering->setStyleSheet("*{color: white;}");
   ui.buttonPup_rest->setStyleSheet("*{color: white;}");
   ui.buttonPup_set->setStyleSheet("*{color: white;}");
   ui.labelPupCh1->setStyleSheet("*{color: white;}");
   ui.pupCh1->setStyleSheet("*{color: white;}");
   ui.labelPupCh2->setStyleSheet("*{color: white;}");
   ui.pupCh2->setStyleSheet("*{color: white;}");
   
   ui.labelx->setStyleSheet("*{color: white;}");
   ui.labely->setStyleSheet("*{color: white;}");
   ui.labelD->setStyleSheet("*{color: white;}");
   ui.labelUR->setStyleSheet("*{color: white;}");
   ui.labelUL->setStyleSheet("*{color: white;}");
   ui.labelLR->setStyleSheet("*{color: white;}");
   ui.labelLL->setStyleSheet("*{color: white;}");
   ui.labelAvg->setStyleSheet("*{color: white;}");
   
   ui.coordUR_x->setStyleSheet("*{color: white;}");
   ui.coordUR_y->setStyleSheet("*{color: white;}");
   ui.coordUR_D->setStyleSheet("*{color: white;}");
   
   ui.coordUL_x->setStyleSheet("*{color: white;}");
   ui.coordUL_y->setStyleSheet("*{color: white;}");
   ui.coordUL_D->setStyleSheet("*{color: white;}");
   
   ui.coordLR_x->setStyleSheet("*{color: white;}");
   ui.coordLR_y->setStyleSheet("*{color: white;}");
   ui.coordLR_D->setStyleSheet("*{color: white;}");
   
   ui.coordLL_x->setStyleSheet("*{color: white;}");
   ui.coordLL_y->setStyleSheet("*{color: white;}");
   ui.coordLL_D->setStyleSheet("*{color: white;}");
   
   ui.coordUR_x->setStyleSheet("*{color: white;}");
   ui.coordUR_y->setStyleSheet("*{color: white;}");
   ui.coordUR_D->setStyleSheet("*{color: white;}");
   
   ui.coordAvg_x->setStyleSheet("*{color: white;}");
   ui.coordAvg_y->setStyleSheet("*{color: white;}");
   ui.coordAvg_D->setStyleSheet("*{color: white;}");
   
   ui.setDelta_pup->setStyleSheet("*{color: white;}");
   ui.button_pup_scale->setStyleSheet("*{color: white;}");
}
   
pupilGuide::~pupilGuide()
{
}

int pupilGuide::subscribe( multiIndiPublisher * publisher )
{
   publisher->subscribeProperty(this, "camwfs-avg", "nAverage");
   
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant1");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant2");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant3");
   publisher->subscribeProperty(this, "camwfs-fit", "quadrant4");
   publisher->subscribeProperty(this, "camwfs-fit", "threshold");
   
   publisher->subscribeProperty(this, "modwfs", "modFrequency");
   publisher->subscribeProperty(this, "modwfs", "modRadius");
   publisher->subscribeProperty(this, "modwfs", "modState");
   publisher->subscribeProperty(this, "modwfs", "fsm");
   
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
   
      if(ipRecv.getName() == "quadrant2")
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
      
      if(ipRecv.getName() == "quadrant3")
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
   
      if(ipRecv.getName() == "quadrant4")
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
      
      if(ipRecv.getName() == "threshold")
      {
         if(ipRecv.find("current"))
         {
            m_threshold_current = ipRecv["current"].get<double>();
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

void pupilGuide::updateGUI()
{
   
   char str[16];
   if(m_modFsmState == "READY")
   {
      if(m_modState == 1)
      {
         ui.modState->setText("RIP");
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
      ui.modState->setText(m_modFsmState.c_str());
   }
   
   snprintf(str,sizeof(str), "%0.2f", m_modCh1);
   ui.modCh1->setText(str);
   snprintf(str,sizeof(str), "%0.2f", m_modCh2);
   ui.modCh2->setText(str);
   
   
   
   if(m_pupFsmState == "READY")
   {
      ui.pupState->setText("SET");
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(true);
   }
   else if(m_pupFsmState == "NOTHOMED")
   {
      ui.pupState->setText("RIP");
      ui.buttonPup_set->setEnabled(true);
      ui.buttonPup_rest->setEnabled(false);
   }
   else if(m_pupFsmState == "HOMING")
   {
      ui.pupState->setText("SETTING");
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(true);
   }
   else
   {
      ui.pupState->setText(m_pupFsmState.c_str());
      ui.buttonPup_set->setEnabled(false);
      ui.buttonPup_rest->setEnabled(false);
   }
   
   snprintf(str,sizeof(str), "%0.2f", m_pupCh1);
   ui.pupCh1->setText(str);
   snprintf(str,sizeof(str), "%0.2f", m_pupCh2);
   ui.pupCh2->setText(str);
   
   if(m_modState == 3)
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
   else if(m_modState == 4)
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
      
      ui.modFreq_target->setEnabled(false);
      ui.modRad_target->setEnabled(false);
      ui.buttonMod_rest->setEnabled(true);
      ui.buttonMod_set->setEnabled(true);
      ui.buttonMod_mod->setEnabled(false);
   }
   
   
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
   ui.med1->setEnabled(true);
   
   snprintf(str, 16, "%0.1f", m2);
   ui.med2->setText(str);
   ui.med2->setEnabled(true);
   
   snprintf(str, 16, "%0.1f", m3);
   ui.med3->setText(str);
   ui.med3->setEnabled(true);
   
   snprintf(str, 16, "%0.1f", m4);
   ui.med4->setText(str);
   ui.med4->setEnabled(true);
   
   snprintf(str, 16, "%0.4f", m_threshold_current);
   ui.fitThreshold_current->setText(str);
   
   snprintf(str, 16, "%u", m_nAverage_current);
   ui.nAverage_current->setText(str);
   
   
   
   
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
   ui.coordLL_D->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", D2);
   ui.coordLR_D->setText(str);
   ui.coordLR_D->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", D3);         
   ui.coordUL_D->setText(str);
   ui.coordUL_D->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", D4);
   ui.coordUR_D->setText(str);
   ui.coordUR_D->setEnabled(true);
   
   
   snprintf(str, 16, "%0.2f", x1);
   ui.coordLL_x->setText(str);
   ui.coordLL_x->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", x2);
   ui.coordLR_x->setText(str);
   ui.coordLR_x->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", x3);
   ui.coordUL_x->setText(str);
   ui.coordUL_x->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", x4);
   ui.coordUR_x->setText(str);
   ui.coordUR_x->setEnabled(true);
   
   
   snprintf(str, 16, "%0.2f", y1);
   ui.coordLL_y->setText(str);
   ui.coordLL_y->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", y2);
   ui.coordLR_y->setText(str);
   ui.coordLR_y->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", y3);
   ui.coordUL_y->setText(str);
   ui.coordUL_y->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", y4);
   ui.coordUR_y->setText(str);
   ui.coordUR_y->setEnabled(true);
   
   
   snprintf(str, 16, "%0.2f", 0.25*(D1+D2+D3+D4));
   ui.coordAvg_D->setText(str);
   ui.coordAvg_D->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", 0.25*(x1+x2+x3+x4));
   ui.coordAvg_x->setText(str);
   ui.coordAvg_x->setEnabled(true);
   
   snprintf(str, 16, "%0.2f", 0.25*(y1+y2+y3+y4));
   ui.coordAvg_y->setText(str);
   ui.coordAvg_y->setEnabled(true);
   
   
   if( (m_camlensxFsmState == "READY" || m_camlensxFsmState == "OPERATING") &&
          (m_camlensyFsmState == "READY" || m_camlensyFsmState == "OPERATING") )
   {
      ui.camlens_fsm->setEnabled(true);
      ui.camlens_x_label->setEnabled(true);
      ui.camlens_x_pos->setEnabled(true);
      ui.camlens_y_label->setEnabled(true);
      ui.camlens_y_pos->setEnabled(true);
      ui.button_camlens_ul->setEnabled(true);
      ui.button_camlens_u->setEnabled(true);
      ui.button_camlens_ur->setEnabled(true);
      ui.button_camlens_l->setEnabled(true);
      ui.button_camlens_r->setEnabled(true);
      ui.button_camlens_dl->setEnabled(true);
      ui.button_camlens_d->setEnabled(true);
      ui.button_camlens_dr->setEnabled(true);
      ui.button_camlens_scale->setEnabled(true);
      
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
      ui.camlens_fsm->setEnabled(false);
      ui.camlens_x_label->setEnabled(false);
      ui.camlens_x_pos->setEnabled(false);
      ui.camlens_y_label->setEnabled(false);
      ui.camlens_y_pos->setEnabled(false);
      ui.button_camlens_ul->setEnabled(false);
      ui.button_camlens_u->setEnabled(false);
      ui.button_camlens_ur->setEnabled(false);
      ui.button_camlens_l->setEnabled(false);
      ui.button_camlens_r->setEnabled(false);
      ui.button_camlens_dl->setEnabled(false);
      ui.button_camlens_d->setEnabled(false);
      ui.button_camlens_dr->setEnabled(false);
      ui.button_camlens_scale->setEnabled(false);
      
      if(m_camlensxFsmState == "HOMING" || m_camlensyFsmState == "HOMING")
      {
         ui.camlens_fsm->setText("HOMING");
      }
      else if(m_camlensxFsmState == "POWERON" || m_camlensyFsmState == "POWERON")
      {
         ui.camlens_fsm->setText("POWERON");
      }
      else if(m_camlensxFsmState == "POWEROFF" && m_camlensyFsmState == "POWEROFF")
      {
         ui.camlens_fsm->setText("POWEROFF");
      }
      else
      {
         ui.camlens_fsm->setText(m_camlensxFsmState.c_str());
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

void pupilGuide::on_buttonScale_pressed()
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
   ui.buttonScale->setText(ss);


}


void pupilGuide::on_button_pup_u_pressed()
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
   ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_l_pressed()
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
   ip["target"] = m_pupCh1 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip);
   
   pcf::IndiProperty ip2(pcf::IndiProperty::Number);
   
   ip2.setDevice("ttmpupil");
   ip2.setName("pos_2");
   ip2.add(pcf::IndiElement("target"));
   ip2["target"] = m_pupCh2 - m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_d_pressed()
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
   ip2["target"] = m_pupCh2 + m_pupStepSize/sqrt(2);
   
   sendNewProperty(ip2);
}

void pupilGuide::on_button_pup_r_pressed()
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
   ip["target"] = m_pupCh1 + m_pupStepSize/sqrt(2);
   
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
   ip["target"] = m_camlensy_pos + m_camlensStepSize;
   
   sendNewProperty(ip);
}

void pupilGuide::on_button_camlens_ul_pressed()
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
   ip["target"] = m_camlensy_pos - sqrt(2)*m_camlensStepSize;
   
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
   ip["target"] = m_camlensy_pos - m_camlensStepSize;
   
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
   ip["target"] = m_camlensy_pos + sqrt(2)*m_camlensStepSize;
   
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
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("releaseDM");
   ip.add(pcf::IndiElement("request"));
   ip["request"] = 1;
   
   sendNewProperty(ip);
   
}

void pupilGuide::on_buttonPup_set_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("ttmpupil");
   ip.setName("initDM");
   ip.add(pcf::IndiElement("request"));
   ip["request"] = 1;
   
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
