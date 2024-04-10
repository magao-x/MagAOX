
#ifndef loopCtrl_hpp
#define loopCtrl_hpp


#include <mutex>
#include "ui_loopCtrl.h"

#include "../xWidgets/xWidget.hpp"
#include "../xWidgets/gainCtrl.hpp"
#include "../xWidgets/statusEntry.hpp"

namespace xqt 
{
   
class loopCtrl : public xWidget
{
   Q_OBJECT
   
protected:
   
   std::string m_procName;
   
   std::string m_loopName;
   std::string m_loopNumber;
   std::string m_gainCtrl;

   std::string m_appState;
   
   double m_gain {0.0};;
   double m_gainScale {0.01};
   
   double m_multcoeff {1};
   double m_multcoeffScale {0.001};
   
   bool m_loopState {false};
   bool m_loopWaiting {false}; //indicates slider is waiting on loop_state update to be re-enabled 
   
   bool m_procState {false};
   
   std::vector<int> m_modes;
   std::vector<gainCtrl *> m_blockCtrls {nullptr};
   std::mutex m_blockMutex;

   QTimer * m_updateTimer {nullptr};

public:
   loopCtrl( std::string & procName,
             QWidget * Parent = 0, 
             Qt::WindowFlags f = Qt::WindowFlags()
           );
   
   ~loopCtrl();
   
   void subscribe();

   void onConnect();

   void onDisconnect();
                                   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleDelProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void sendNewGain(double ng);
   void sendNewMultCoeff(double nm);
   
   void setEnableDisable( bool tf, 
                          bool all = true
                        );

public slots:
   void updateGUI();
      
   //void on_slider_loop_sliderReleased();
   
   void on_button_LoopZero_pressed();
   
   void on_button_zeroall_pressed();

   void setupBlocks(int nB);

   void on_button_setplaw_pressed();

signals:

   void doUpdateGUI();

   void blocksChanged(int nB);

private:
     
   Ui::loopCtrl ui;
};
   
loopCtrl::loopCtrl( std::string & procName,
                    QWidget * Parent, 
                    Qt::WindowFlags f) : xWidget(Parent, f), m_procName{procName}
{
   ui.setupUi(this);
   
   connect(this, SIGNAL(blocksChanged(int)), this, SLOT(setupBlocks(int)));

   setWindowTitle(QString(m_procName.c_str()));
   ui.label_loop_state->setProperty("isStatus", true);

   ui.slider_loop->setup(m_procName, "loop_state", "toggle", "");
   ui.slider_loop->setStretch(0,0,10, true, true);

   ui.gainCtrl->setup(m_procName, "loop_gain", "Gain", -1, -1);
   ui.mcCtrl->setup(m_procName, "loop_multcoeff", "Mult. Coef.", -1, -1);
   ui.mcCtrl->makeMultCoeffCtrl();

   if(m_procName == "loloop") m_gainCtrl = "logainctrl";
   else if(m_procName == "wloop") m_gainCtrl = "wgainctrl";
   else m_gainCtrl = "hogainctrl";

   ui.powerLawIndex->setup(m_gainCtrl, "pwrlaw_index", statusEntry::FLOAT, "Index", "");
   ui.powerLawFloor->setup(m_gainCtrl, "pwrlaw_floor", statusEntry::FLOAT, "Floor", "");

   setXwFont(ui.label_LoopName);
   setXwFont(ui.label_loop);
   setXwFont(ui.label_loop_state);
   setXwFont(ui.button_LoopZero);
   setXwFont(ui.button_zeroall);
   setXwFont(ui.label_block_gains);

   setXwFont(ui.label_powerLaw);

   m_updateTimer = new QTimer;

   connect(m_updateTimer, SIGNAL(timeout()), this, SLOT(updateGUI()));

   m_updateTimer->start(250);

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
   
   m_parent->addSubscriber(ui.slider_loop);
   m_parent->addSubscriberProperty(this, m_procName, "loop_state");

   m_parent->addSubscriber(ui.powerLawIndex);
   m_parent->addSubscriber(ui.powerLawFloor);

   m_parent->addSubscriberProperty(this, m_gainCtrl, "modes");
   

   m_parent->addSubscriber(ui.gainCtrl);
   m_parent->addSubscriber(ui.mcCtrl);

   std::lock_guard<std::mutex> lock(m_blockMutex);
   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_parent->addSubscriber(m_blockCtrls[n]);
   }

   return;
}
   
void loopCtrl::onConnect()
{
   setWindowTitle(QString(m_procName.c_str()));

   xWidget::onConnect();
   ui.gainCtrl->onConnect();
   ui.mcCtrl->onConnect();
   
   std::lock_guard<std::mutex> lock(m_blockMutex);
   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_blockCtrls[n]->onConnect();
   }
}

void loopCtrl::onDisconnect()
{
   std::string tit = m_procName + " (disconnected)";
   setWindowTitle(QString(tit.c_str()));

   setEnableDisable(false);

   xWidget::onDisconnect();
   ui.gainCtrl->onDisconnect();
   ui.mcCtrl->onDisconnect();

   std::lock_guard<std::mutex> lock(m_blockMutex);
   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_blockCtrls[n]->onDisconnect();
   }
}

void loopCtrl::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void loopCtrl::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_procName && ipRecv.getDevice() != m_gainCtrl) return;
   
   if(ipRecv.getDevice() == m_procName)
   {
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
   }
   if(ipRecv.getDevice() == m_gainCtrl)
   {
      if(ipRecv.getName() == "modes")
      {
         if(ipRecv.find("blocks"))
         {
            size_t nB = ipRecv["blocks"].get<int>();

            m_modes.resize(nB,0);
            
            for(size_t n = 0; n<nB; ++n)
            {
               char mstr[24];
               snprintf(mstr, sizeof(mstr), "%02zu", n);
               std::string blockstr = std::string("block")+mstr;
               int nM = ipRecv[std::string("block")+mstr].get<int>();
               m_modes[n] = nM;

            }

            if(nB != m_blockCtrls.size()) emit blocksChanged(nB);

            
         }
      }
   }

   emit doUpdateGUI();
}

void loopCtrl::handleDelProperty( const pcf::IndiProperty & ipRecv)
{  
   std::lock_guard<std::mutex> lock(m_blockMutex);
   
   if(ipRecv.getDevice() == m_gainCtrl)
   {
      for(size_t n =0; n < m_blockCtrls.size(); ++n)
      {
         m_parent->unsubscribe(m_blockCtrls[n]);
         ui.horizontalLayout_2->removeWidget(m_blockCtrls[n]);
         m_blockCtrls[n]->deleteLater();
      }
         
      m_blockCtrls.clear();
   }

}

void loopCtrl::setEnableDisable( bool tf,
                                 bool all
                               )
{
   if(all)
   {
      ui.slider_loop->setEnabled(tf);
      ui.label_loop_state->setEnabled(tf);
   }

   ui.gainCtrl->setEnabled(tf);
   ui.mcCtrl->setEnabled(tf);

   ui.powerLawIndex->setEnabled(tf);
   ui.powerLawFloor->setEnabled(tf);
   ui.button_setplaw->setEnabled(tf);

   std::lock_guard<std::mutex> lock(m_blockMutex);
   for(size_t n = 0; n < m_blockCtrls.size(); ++n)
   {
      if(m_blockCtrls[n]) m_blockCtrls[n]->setEnabled(tf);
   }
}

void loopCtrl::updateGUI()
{      
      setEnableDisable(true, false);

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
   
   
} //updateGUI()

void loopCtrl::on_button_LoopZero_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_procName);
   ipFreq.setName("loop_zero");
   ipFreq.add(pcf::IndiElement("request"));
   
   ipFreq["request"] = pcf::IndiElement::On;
   
   sendNewProperty(ipFreq);
}

void loopCtrl::on_button_zeroall_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_gainCtrl);
   ipFreq.setName("zero_all");
   ipFreq.add(pcf::IndiElement("request"));
   
   ipFreq["request"] = pcf::IndiElement::On;
   
   sendNewProperty(ipFreq);
}

void loopCtrl::setupBlocks(int nB)
{
   std::lock_guard<std::mutex> lock(m_blockMutex);

   m_blockCtrls.resize(nB, nullptr); //I think this will call the destructor

   int modeTot = 0;
   for(int n = 0; n < nB; ++n)
   {
      char str[16];
      snprintf(str, sizeof(str), "%02d", n);
      modeTot += m_modes[n];
      m_blockCtrls[n] = new gainCtrl(m_gainCtrl, std::string("block") + str + "_gain", "", m_modes[n], modeTot);
      ui.horizontalLayout_2->addWidget(m_blockCtrls[n]);
      if(m_parent) m_parent->addSubscriber(m_blockCtrls[n]);
   }
}

void loopCtrl::on_button_setplaw_pressed()
{
    pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
    ipFreq.setDevice(m_gainCtrl);
    ipFreq.setName("pwrlaw_set");
    ipFreq.add(pcf::IndiElement("request"));
   
    ipFreq["request"] = pcf::IndiElement::On;
   
    sendNewProperty(ipFreq);
}

} //namespace xqt
   
#include "moc_loopCtrl.cpp"

#endif
