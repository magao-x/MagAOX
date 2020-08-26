#ifndef coronAlign_hpp
#define coronAlign_hpp

#include <cmath>
#include <unistd.h>

#include <QWidget>
#include <QMutex>
#include <QTimer>

#include "ui_coronAlign.h"

#include "../../lib/multiIndi.hpp"


namespace xqt 
{
   
class coronAlign : public QDialog, public multiIndiSubscriber
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
   
   int m_pupilScale {1};

   //Focal Plane
   
   std::string m_fwFocalState;
   
   
   double m_fwFocalPos;
   long m_picoFocalPos;
   
   int m_focalScale {1};
   
   //Lyot Plane
   
   std::string m_fwLyotState;
   
   
   double m_fwLyotPos;
   long m_picoLyotPos;
   
   int m_lyotScale {1};
   
public:
   coronAlign( QWidget * Parent = 0, Qt::WindowFlags f = 0);
   
   ~coronAlign();
   
   int subscribe( multiIndiPublisher * publisher );
                                   
   virtual void onDisconnect();
   
   int handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   int handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   
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

coronAlign::coronAlign( QWidget * Parent, Qt::WindowFlags f) : QDialog(Parent, f)
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
   
}
   
coronAlign::~coronAlign()
{
}

int coronAlign::subscribe( multiIndiPublisher * publisher )
{
   //publisher->subscribeProperty(this, "modwfs", "modFrequency");
   
   return 0;
}
   
void coronAlign::onDisconnect()
{
   m_picoState = "";
   m_fwPupilState = "";
   m_fwFocalState = "";
   m_fwLyotState = "";
}
   
int coronAlign::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   std::string dev = ipRecv.getDevice();
   if( dev == "picomotors" || 
       dev == "fwpupil" || 
       dev == "fwlyot" || 
       dev == "fwfpm" ) 
   {
      return handleSetProperty(ipRecv);
   }
   
   return 0;
}

int coronAlign::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   std::string dev = ipRecv.getDevice();
   
   return 0;
   //m_mutex.lock();
/*   if(dev == "camwfs-avg")
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
   */
}

void coronAlign::updateGUI()
{
   
   //--------- Modulation 
   /*
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
      ui.labelMedianFluxes->setEnabled(false);
      ui.med1->setEnabled(false);
      ui.med2->setEnabled(false);
      ui.med3->setEnabled(false);
      ui.med4->setEnabled(false);
      ui.setDelta->setEnabled(false);
      ui.labelThreshCurrent->setEnabled(false);
      ui.labelThreshTarget->setEnabled(false);
      ui.labelThreshold->setEnabled(false);
      ui.fitThreshold_current->setEnabled(false);
      ui.fitThreshold_target->setEnabled(false);
      
      ui.med1->setText("");
      ui.med2->setText("");
      ui.med3->setText("");
      ui.med4->setText("");
      ui.fitThreshold_current->setText("");
      
      ui.coordLL_D->setEnabled(false);
      ui.coordLR_D->setEnabled(false);
      ui.coordUL_D->setEnabled(false);
      ui.coordUR_D->setEnabled(false);
      ui.coordLL_x->setEnabled(false);
      ui.coordLR_x->setEnabled(false);
      ui.coordUL_x->setEnabled(false);
      ui.coordUR_x->setEnabled(false);
      ui.coordLL_y->setEnabled(false);
      ui.coordLR_y->setEnabled(false);
      ui.coordUL_y->setEnabled(false);
      ui.coordUR_y->setEnabled(false);
      ui.coordAvg_D->setEnabled(false);
      ui.coordAvg_x->setEnabled(false);
      ui.coordAvg_y->setEnabled(false);
      ui.setDelta_pup->setEnabled(false);
      ui.labelx->setEnabled(false);
      ui.labely->setEnabled(false);
      ui.labelD->setEnabled(false);
      ui.labelUR->setEnabled(false);
      ui.labelUL->setEnabled(false);
      ui.labelLR->setEnabled(false);
      ui.labelLL->setEnabled(false);
      ui.labelAvg->setEnabled(false);
   }
   else
   {
      ui.labelMedianFluxes->setEnabled(true);
      ui.med1->setEnabled(true);
      ui.med2->setEnabled(true);
      ui.med3->setEnabled(true);
      ui.med4->setEnabled(true);
      ui.setDelta->setEnabled(true);
      ui.labelThreshCurrent->setEnabled(true);
      ui.labelThreshTarget->setEnabled(true);
      ui.labelThreshold->setEnabled(true);
      ui.fitThreshold_current->setEnabled(true);
      ui.fitThreshold_target->setEnabled(true);
   
      ui.coordLL_D->setEnabled(true);
      ui.coordLR_D->setEnabled(true);
      ui.coordUL_D->setEnabled(true);
      ui.coordUR_D->setEnabled(true);
      ui.coordLL_x->setEnabled(true);
      ui.coordLR_x->setEnabled(true);
      ui.coordUL_x->setEnabled(true);
      ui.coordUR_x->setEnabled(true);
      ui.coordLL_y->setEnabled(true);
      ui.coordLR_y->setEnabled(true);
      ui.coordUL_y->setEnabled(true);
      ui.coordUR_y->setEnabled(true);
      ui.coordAvg_D->setEnabled(true);
      ui.coordAvg_x->setEnabled(true);
      ui.coordAvg_y->setEnabled(true);
      ui.setDelta_pup->setEnabled(true);
      ui.labelx->setEnabled(true);
      ui.labely->setEnabled(true);
      ui.labelD->setEnabled(true);
      ui.labelUR->setEnabled(true);
      ui.labelUL->setEnabled(true);
      ui.labelLR->setEnabled(true);
      ui.labelLL->setEnabled(true);
      ui.labelAvg->setEnabled(true);

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
      */
} //updateGUI()

void coronAlign::on_button_pupil_u_pressed()
{
   std::cerr << "pupil u\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwpupil");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_pupil_d_pressed()
{
   std::cerr << "pupil d\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwpupil");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_pupil_l_pressed()
{
   std::cerr << "pupil l\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwpupil");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_pupil_r_pressed()
{
   std::cerr << "pupil r\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwpupil");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
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
   std::cerr << "focal u\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwfocal");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_focal_d_pressed()
{
   std::cerr << "focal d\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwfocal");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_focal_l_pressed()
{
   std::cerr << "focal l\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwfocal");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_focal_r_pressed()
{
   std::cerr << "focal r\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwfocal");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
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
   std::cerr << "lyot u\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwlyot");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_lyot_d_pressed()
{
   std::cerr << "lyot d\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwlyot");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_lyot_l_pressed()
{
   std::cerr << "lyot l\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwlyot");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
}

void coronAlign::on_button_lyot_r_pressed()
{
   std::cerr << "lyot r\n";
   /*
   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice("fwlyot");
   ip.setName("offset");
   ip.add(pcf::IndiElement("y"));
   ip["y"] = m_stepSize;
   ip.add(pcf::IndiElement("x"));
   ip["x"] = 0;
   
   sendNewProperty(ip);
   */
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
