
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
   
private:
     
   Ui::pupilGuide ui;
};

} //namespace xqt
   
