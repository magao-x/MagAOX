
#ifndef loopCtrl_hpp
#define loopCtrl_hpp


#include "ui_loopCtrl.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt 
{
   
class loopCtrl : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_procName;
   
   std::string m_loopName;
   std::string m_loopNumber;
   
   std::string m_appState;
   
   double m_gain {0.0};;
   double m_gainScale {0.01};
   
   double m_multcoeff {1};
   double m_multcoeffScale {0.001};
   
   bool m_loopState {false};
   bool m_loopWaiting {false}; //indicates slider is waiting on loop_state update to be re-enabled 
   
   bool m_procState {false};
   
public:
   loopCtrl( std::string & procName,
             QWidget * Parent = 0, 
             Qt::WindowFlags f = 0
           );
   
   ~loopCtrl();
   
   void subscribe();
                                   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void sendNewGain(double ng);
   void sendNewMultCoeff(double nm);
   
public slots:
   void updateGUI();
   
   void on_button_gainScale_pressed();
   void on_button_gainUp_pressed();
   void on_button_gainDown_pressed();
   void on_lineEdit_gain_returnPressed();
   void on_button_zeroGain_pressed();
   
   void on_button_multcoeffScale_pressed();
   void on_button_multcoeffUp_pressed();
   void on_button_multcoeffDown_pressed();
   void on_lineEdit_multcoeff_returnPressed();
   void on_button_oneMultCoeff_pressed();
   
   void on_slider_loop_sliderReleased();
   
   void on_button_LoopZero_pressed();
   
private:
     
   Ui::loopCtrl ui;
};
   
loopCtrl::loopCtrl( std::string & procName,
                    QWidget * Parent, 
                    Qt::WindowFlags f) : xWidget(Parent, f), m_procName{procName}
{
   ui.setupUi(this);
   
   setWindowTitle(QString(m_procName.c_str()));
   ui.button_gainScale->setText(QString::number(m_gainScale));
   ui.label_loop_state->setProperty("isStatus", true);
   ui.lcd_gain->setProperty("isStatus", true);
   ui.lcd_multcoeff->setProperty("isStatus", true);

   onDisconnect();
}
   
loopCtrl::~loopCtrl()
{
}

void loopCtrl::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty(this, m_procName, "fsm");
   m_parent->addSubscriberProperty(this, m_procName, "loop");
   m_parent->addSubscriberProperty(this, m_procName, "loop_gain");
   m_parent->addSubscriberProperty(this, m_procName, "loop_multcoeff");
   m_parent->addSubscriberProperty(this, m_procName, "loop_processes");
   m_parent->addSubscriberProperty(this, m_procName, "loop_state");
   
   return;
}
   
void loopCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void loopCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_procName) return;
   
   if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }
   else if(ipRecv.getName() == "loop")
   {
      if(ipRecv.find("name"))
      {
         m_loopName = ipRecv["name"].get<std::string>();
      }
      
      if(ipRecv.find("number"))
      {
         m_loopNumber = ipRecv["number"].get<std::string>();
      }
      
      std::string label = m_loopName + " (aol" + m_loopNumber + ")";
      ui.label_LoopName->setText(label.c_str());
   }
   else if(ipRecv.getName() == "loop_gain")
   {
      if(ipRecv.find("current"))
      {
         m_gain = ipRecv["current"].get<float>();
      }
   }
   else if(ipRecv.getName() == "loop_multcoeff")
   {
      if(ipRecv.find("current"))
      {
         m_multcoeff = ipRecv["current"].get<float>();
      }
   }
   else if(ipRecv.getName() == "loop_state")
   {
      if(ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
         {
            m_loopState = true;
         }
         else
         {
            m_loopState = false;
         }
         
         m_loopWaiting = false;
      }
   }
   else if(ipRecv.getName() == "loop_processes")
   {
      if(ipRecv.find("toggle"))
      {
         if(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On)
         {
            m_procState = true;
         }
         else
         {
            m_procState = false;
         }
      }
   }
   
   updateGUI();
}

void loopCtrl::sendNewGain(double ng)
{
   if(ng < 0)
   {
      std::cerr << "loopCtrl: requested gain out of range, can't be < 0";
      
      return;
   }
   
   try
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);

      ipFreq.setDevice(m_procName);
      ipFreq.setName("loop_gain");
      ipFreq.add(pcf::IndiElement("current"));
      ipFreq.add(pcf::IndiElement("target"));
      ipFreq["current"] = ng;
      ipFreq["target"] = ng;
   
      sendNewProperty(ipFreq);   
   }
   catch(...)
   {
      std::cerr << "libcommon INDI exception.  going on. (" << __FILE__ << " " << __LINE__ << "\n";
   }
}

void loopCtrl::sendNewMultCoeff(double nm)
{
   if(nm < 0)
   {
      std::cerr << "loopCtrl: requested mult. coeff. out of range, can't be < 0";
      
      nm = 0;
   }
   
   if(nm > 1)
   {
      std::cerr << "loopCtrl: requested mult. coeff. out of range, can't be > 1";
      
      nm = 1;
   }
   
   try
   {
      pcf::IndiProperty ipFreq(pcf::IndiProperty::Number);

      ipFreq.setDevice(m_procName);
      ipFreq.setName("loop_multcoeff");
      ipFreq.add(pcf::IndiElement("current"));
      ipFreq.add(pcf::IndiElement("target"));
      ipFreq["current"] = nm;
      ipFreq["target"] = nm;
   
      sendNewProperty(ipFreq);   
   }
   catch(...)
   {
      std::cerr << "libcommon INDI exception.  going on. (" << __FILE__ << " " << __LINE__ << "\n";
   }
}

void loopCtrl::updateGUI()
{
   ui.lcd_gain->display(m_gain);
   ui.lcd_multcoeff->display(m_multcoeff);
   
   if(!m_loopWaiting)
   {
      if(m_loopState)
      {
         ui.slider_loop->setSliderPosition(ui.slider_loop->maximum());
      }
      else
      {
         ui.slider_loop->setSliderPosition(ui.slider_loop->minimum());
      }
   }
   
   if( m_appState != "READY" && m_appState != "OPERATING" )
   {
      /// \todo Disable & zero all
      
      return;
   }
   
   if(!m_procState)
   {
      ui.slider_loop->setEnabled(false);
      ui.label_loop_state->setText("processes off");
      ui.label_loop_state->setEnabled(false);
      ui.lcd_gain->setEnabled(false);
      ui.lcd_multcoeff->setEnabled(false);
   }
   else
   {
      ui.lcd_gain->setEnabled(true);
      ui.lcd_multcoeff->setEnabled(true);
      
      if(!m_loopWaiting) 
      {
         ui.label_loop_state->setEnabled(true);
         ui.slider_loop->setEnabled(true);
      }
      
      if(m_loopState)
      {
         ui.label_loop_state->setText("CLOSED");
      }
      else
      {
         ui.label_loop_state->setText("OPEN");
      }
   }
   
} //updateGUI()



void loopCtrl::on_button_gainScale_pressed()
{
   //Can only be 0.01, 0.05, or 0.1, but we make sure floating point doesn't scew us up.  
   //the progresion is:
   // 0.01->0.1->0.05->0.01
   if(m_gainScale < 0.05) //0.01
   {
      m_gainScale = 0.1;
   }
   else if(m_gainScale < 0.1) //0.05
   {
      m_gainScale = 0.01;
   }
   else //0.1
   {
      m_gainScale = 0.05;
   }
   
   ui.button_gainScale->setText(QString::number(m_gainScale));
}

void loopCtrl::on_button_gainUp_pressed()
{
   double ng = m_gain + m_gainScale;
   sendNewGain(ng);
}

void loopCtrl::on_button_gainDown_pressed()
{
   double ng = m_gain - m_gainScale;
   if(ng < 0.00001) ng = 0; //Stop floating point nonsense
   sendNewGain(ng);
}

void loopCtrl::on_lineEdit_gain_returnPressed()
{
   double ng = ui.lineEdit_gain->text().toDouble();
   
   if(ng < 0) ng = 0;

   sendNewGain(ng);
   
   ui.lineEdit_gain->setText("");
}

void loopCtrl::on_button_zeroGain_pressed()
{
   sendNewGain(0.0);
}

void loopCtrl::on_button_multcoeffScale_pressed()
{
   //Can only be 0.001, 0.002, 0.005, or 0.01 but we make sure floating point doesn't scew us up.  
   //the progresion is:
   // 0.001->0.01->0.005->0.002->0.001
   if(m_multcoeffScale < 0.002) //0.001
   {
      m_multcoeffScale = 0.01;
   }
   else if(m_multcoeffScale < 0.005) //0.002
   {
      m_multcoeffScale = 0.001;
   }
   else if(m_multcoeffScale < 0.01) //0.005
   {
      m_multcoeffScale = 0.002;
   }
   else //0.01
   {
      m_multcoeffScale = 0.005;
   }
   
   ui.button_multcoeffScale->setText(QString::number(m_multcoeffScale));
}

void loopCtrl::on_button_multcoeffUp_pressed()
{
   double nm = m_multcoeff + m_multcoeffScale;
   if(nm > 1) nm = 1;
   sendNewMultCoeff(nm);
}

void loopCtrl::on_button_multcoeffDown_pressed()
{
   double nm = m_multcoeff - m_multcoeffScale;
   if(nm < 0.00001) nm = 0; //Stop floating point nonsense
   sendNewMultCoeff(nm);
}

void loopCtrl::on_lineEdit_multcoeff_returnPressed()
{
   double nm = ui.lineEdit_multcoeff->text().toDouble();
   
   if(nm < 0) nm = 0;
   if(nm > 1) nm = 1;
   
   sendNewMultCoeff(nm);
   
   ui.lineEdit_multcoeff->setText("");
}

void loopCtrl::on_button_oneMultCoeff_pressed()
{
   sendNewMultCoeff(1.0);
}

void loopCtrl::on_slider_loop_sliderReleased()
{
   double relpos = ((double)(ui.slider_loop->sliderPosition() - ui.slider_loop->minimum()))/(ui.slider_loop->maximum() - ui.slider_loop->minimum());
   
   if(relpos > 0.1 && relpos < 0.9)
   {
      if(m_loopState)
      {
         ui.slider_loop->setSliderPosition(ui.slider_loop->maximum());
      }
      else
      {
         ui.slider_loop->setSliderPosition(ui.slider_loop->minimum());
      }
      return;
   }
   
   ui.label_loop_state->setText("-----");
   ui.label_loop_state->setEnabled(false);
   ui.slider_loop->setEnabled(false);
   m_loopWaiting = true;
   
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_procName);
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

void loopCtrl::on_button_LoopZero_pressed()
{
   std::cerr << "loop zero\n";
}

} //namespace xqt
   
#include "moc_loopCtrl.cpp"

#endif
