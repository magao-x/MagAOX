
#ifndef stage_hpp
#define stage_hpp

#include "ui_stage.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt 
{
   
class stage : public xWidget
{
   Q_OBJECT
   
   enum editchanges{NOTEDITING, STARTED, STOPPED};

protected:
   
   std::string m_appState;
   
   std::string m_stageName;
   std::string m_winTitle;

   std::vector<std::string> m_presets;
   std::string m_presetCurrent;
   std::string m_presetTarget;

   bool m_filterWheel {false};

   std::string m_setPoint;

   double m_maxPos  {100};
   double m_position {-1e30};
   bool m_position_changed {false};

   double m_step {1};

   int m_setPointEditing {STOPPED};
   QTimer * m_setPointEditTimer {nullptr};

public:
   explicit stage( std::string & stageName,
                   QWidget * Parent = 0, 
                   Qt::WindowFlags f = Qt::WindowFlags()
                 );
   
   ~stage();
   
   void subscribe();
             
   virtual void onConnect();
   virtual void onDisconnect();
   
   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);
   
   void clear_focus();

public slots:
   void updateGUI();
   
   void on_setPoint_activated( int index );

   void on_setPointGo_pressed();

   void on_positionSlider_sliderMoved( double s );

   void on_positionSlider_sliderReleased();

   void on_position_returnPressed();

   void on_stepSize_editingFinished();

   void on_posMinus_pressed();
   void on_posPlus_pressed();

   void on_posStepMulTen_pressed();
   void on_posStepDivTen_pressed();

   void on_home_pressed();
   void on_stop_pressed();
      
   void setPointEditTimerOut();

signals:
   void setPointEditTimerStart(int);

   void doUpdateGUI();

protected:
   virtual void paintEvent(QPaintEvent * e);

private:
     
   Ui::stage ui;
};
   
stage::stage( std::string & stageName,
              QWidget * Parent, 
              Qt::WindowFlags f) : xWidget(Parent, f), m_stageName{stageName}
{
    ui.setupUi(this);

    m_winTitle = m_stageName;
   
    ui.fsmState->device(m_stageName);
    QFont qf = ui.stageName->font();
    qf.setPixelSize(XW_FONT_SIZE+3);
    ui.stageName->setFont(qf);

    ui.stageName->setText(m_stageName.c_str());

    ui.setPoint->setProperty("isStatus", true);

    ui.position->setAlignment(Qt::AlignCenter);
    ui.stepSize->setAlignment(Qt::AlignCenter);

    m_setPointEditTimer = new QTimer(this);
    connect(m_setPointEditTimer, SIGNAL(timeout()), this, SLOT(setPointEditTimerOut()));
    connect(this, SIGNAL(setPointEditTimerStart(int)), m_setPointEditTimer, SLOT(start(int)));
    connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

    onDisconnect();
}
   
stage::~stage()
{
}

void stage::subscribe()
{
   if(!m_parent) return;
   
   m_parent->addSubscriberProperty(this, m_stageName, "fsm");
   m_parent->addSubscriberProperty(this, m_stageName, "maxpos");
   m_parent->addSubscriberProperty(this, m_stageName, "position");
   m_parent->addSubscriberProperty(this, m_stageName, "filter");
   m_parent->addSubscriberProperty(this, m_stageName, "presetName");
   m_parent->addSubscriberProperty(this, m_stageName, "filterName");
   m_parent->addSubscriber(ui.fsmState);

   return;
}
  
void stage::onConnect()
{
   setWindowTitle(QString(m_winTitle.c_str()));

   ui.stepSize->setText(QString::number(m_step));

   clearFocus();
}

void stage::onDisconnect()
{
   setWindowTitle(QString(m_winTitle.c_str()) + QString(" (disconnected)"));

   ui.stageName->setEnabled(false);
   ui.fsmState->setEnabled(false);
   ui.setPoint->setEnabled(false);
   ui.setPointGo->setEnabled(false);
   ui.positionSlider->setEnabled(false);
   ui.position->setText("---");
   ui.position->setEnabled(false);
   ui.stepSize->setEnabled(false);

   ui.posMinus->setEnabled(false);
   ui.posPlus->setEnabled(false);
   ui.posStepMulTen->setEnabled(false);
   ui.posStepDivTen->setEnabled(false);

   ui.home->setEnabled(false);
   ui.stop->setEnabled(false);

   ui.fsmState->onDisconnect();
}

void stage::handleDefProperty( const pcf::IndiProperty & ipRecv)
{  
   return handleSetProperty(ipRecv);
}

void stage::handleSetProperty( const pcf::IndiProperty & ipRecv)
{  
   if(ipRecv.getDevice() != m_stageName) return;
   
   if(ipRecv.getName() == "fsm")
   {
      if(ipRecv.find("state"))
      {
         m_appState = ipRecv["state"].get<std::string>();
      }
   }

   if(ipRecv.getName() == "maxpos")
   {
      if(ipRecv.find("value"))
      {
         m_maxPos = ipRecv["value"].get<double>();
      }
   }

   if(ipRecv.getName()== "position")
   {
      if(ipRecv.find("current"))
      {
         double val = ipRecv["current"].get<double>();
         if(val != m_position) m_position_changed = true;
         m_position = val;
      }
   }
   else if(ipRecv.getName()== "filter")
   {
      m_filterWheel = true;
      if(ipRecv.find("current"))
      {
         double val = ipRecv["current"].get<double>();
         if(val != m_position) m_position_changed = true;
         m_position = val;
      }
   }

   if(ipRecv.getName() == "presetName" || ipRecv.getName() == "filterName")
   {
      if(ipRecv.getName() == "filterName") m_filterWheel = true;

      int n =0;
      std::string newName;
      for(auto it = ipRecv.getElements().begin(); it != ipRecv.getElements().end(); ++it)
      {
         ++n;
         if(ui.setPoint->findText(QString(it->second.name().c_str())) == -1)
         {
            ui.setPoint->addItem(QString(it->second.name().c_str()));
         }

         if(it->second.getSwitchState() == pcf::IndiElement::On)
         {
            if(newName != "")
            {
               std::cerr << "More than one switch selected in " << ipRecv.getDevice() << "." << ipRecv.getName() << "\n";
            }
         
            newName = it->second.name();
            m_setPoint = newName;
            ui.setPoint->setCurrentText(m_setPoint.c_str());
            //if(newName != m_value) m_valChanged = true;
            //m_value = newName; 
         }
      }

      if(m_filterWheel) m_maxPos = n + 0.5;
   }

   emit doUpdateGUI();
}

void stage::updateGUI()
{
   if( m_appState != "READY" && m_appState != "OPERATING" 
             && m_appState != "CONFIGURING"  && m_appState != "NOTHOMED" 
                     && m_appState != "HOMING")
   {
      ui.stageName->setEnabled(false);
      ui.fsmState->setEnabled(false);
      ui.setPoint->setEnabled(false);
      ui.setPointGo->setEnabled(false);
      ui.positionSlider->setEnabled(false);
      ui.position->setText("---");
      ui.position->setEnabled(false);
      ui.stepSize->setEnabled(false);
   
      ui.posMinus->setEnabled(false);
      ui.posPlus->setEnabled(false);
      ui.posStepMulTen->setEnabled(false);
      ui.posStepDivTen->setEnabled(false);
   
      ui.home->setEnabled(false);
      ui.stop->setEnabled(false);

      return;
   }
   
   if( m_appState == "READY" || m_appState == "OPERATING" || m_appState == "HOMING" || m_appState == "CONFIGURING")
   {
      ui.stageName->setEnabled(true);
      ui.fsmState->setEnabled(true);
      ui.setPoint->setEnabled(true);
      ui.stepSize->setEnabled(true);
      ui.posStepMulTen->setEnabled(true);
      ui.posStepDivTen->setEnabled(true);
      ui.stop->setEnabled(true);
   
      if( m_appState == "READY" )
      {
         ui.setPointGo->setEnabled(true);
         ui.positionSlider->setEnabled(true);
         ui.position->setEnabled(true);
         ui.posMinus->setEnabled(true);
         ui.posPlus->setEnabled(true);
         ui.home->setEnabled(true);
      }
      else  
      {
         ui.setPointGo->setEnabled(false);
         ui.positionSlider->setEnabled(false);
         ui.position->setEnabled(false);
         ui.posMinus->setEnabled(false);
         ui.posPlus->setEnabled(false);
         ui.home->setEnabled(false);
      }
   }
   else if( m_appState == "NOTHOMED")
   {
      ui.stageName->setEnabled(true);
      ui.fsmState->setEnabled(true);
      ui.setPoint->setEnabled(false);
      ui.stepSize->setEnabled(false);
      ui.posStepMulTen->setEnabled(false);
      ui.posStepDivTen->setEnabled(false);
      ui.stop->setEnabled(false);

      ui.setPointGo->setEnabled(false);
      ui.positionSlider->setEnabled(false);
      ui.position->setEnabled(false);
      ui.posMinus->setEnabled(false);
      ui.posPlus->setEnabled(false);
      ui.home->setEnabled(true);
   }

   if(m_position_changed)
   {
      ui.position->setTextChanged(QString::number(m_position));
      m_position_changed = false;
   }
   else
   {
      ui.position->setText(QString::number(m_position));
   }

   ui.positionSlider->setValue(m_position/m_maxPos * 100.);
   //ui.position->updateGUI();

} //updateGUI()

void stage::clear_focus()
{
}

void stage::on_setPoint_activated(int index)
{
   static_cast<void>(index);

   m_setPointEditing = STARTED;
   emit setPointEditTimerStart(10000);
   update();
}

void stage::on_setPointGo_pressed()
{
   std::string selection = ui.setPoint->currentText().toStdString();

   if(selection == "")
   {
      return;
   }

   try
   {
      pcf::IndiProperty ipSend(pcf::IndiProperty::Switch);
      ipSend.setDevice(m_stageName);
      if(m_filterWheel)
      {
         ipSend.setName("filterName");
      }
      else
      {
         ipSend.setName("presetName");
      }
      ipSend.setPerm(pcf::IndiProperty::ReadWrite); 
      ipSend.setState(pcf::IndiProperty::Idle);
      ipSend.setRule(pcf::IndiProperty::OneOfMany);
   
      for(int idx = 0; idx < ui.setPoint->count(); ++idx)
      {
         std::string elName = ui.setPoint->itemText(idx).toStdString();
   
         if(elName == selection)
         {
            ipSend.add(pcf::IndiElement(elName, pcf::IndiElement::On));
         }
         else
         {
            ipSend.add(pcf::IndiElement(elName, pcf::IndiElement::Off));
         }
      }
   
      sendNewProperty(ipSend);
   }
   catch(...)
   {
      std::cerr << "exception thrown in stage::on_setPointGo_pressed\n";
   }

   m_setPointEditing = STOPPED;
   ui.setPoint->setCurrentText(m_setPoint.c_str());
   update();
}

void stage::on_positionSlider_sliderMoved( double s )
{
   double epos = s/100.0 * m_maxPos;

   ui.position->setEditText(QString::number(epos));  
}

void stage::on_positionSlider_sliderReleased()
{
   static double lastPos {-1e30};

   ui.position->stopEditing();

   double s = ui.positionSlider->value();
   double newPos = s/100.0 * m_maxPos;

   ui.positionSlider->setValue(m_position/m_maxPos * 100.);

   if(newPos == lastPos) return;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_stageName);
   if(m_filterWheel)
   {
      ip.setName("filter");
   }
   else
   {
      ip.setName("position");
   }
   ip.add(pcf::IndiElement("target"));
   ip["target"] = newPos;

   lastPos = newPos;

   sendNewProperty(ip);

}

void stage::on_position_returnPressed()
{
   static double lastPos {-1e30};

   double newPos = ui.position->editText().toDouble();

   if(newPos == lastPos) return;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_stageName);
   if(m_filterWheel)
   {
      ip.setName("filter");
   }
   else
   {
      ip.setName("position");
   }
   ip.add(pcf::IndiElement("target"));
   ip["target"] = newPos;
   lastPos = newPos;
   sendNewProperty(ip);

   ui.position->clearFocus();
}

void stage::on_stepSize_editingFinished()
{
   m_step = ui.stepSize->text().toDouble();
   ui.stepSize->setText(QString::number(m_step));

}

void stage::on_posMinus_pressed()
{
   double newPos = m_position - m_step;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_stageName);
   if(m_filterWheel)
   {
      ip.setName("filter");
   }
   else
   {
      ip.setName("position");
   }
   ip.add(pcf::IndiElement("target"));
   ip["target"] = newPos;
   
   sendNewProperty(ip);
}

void stage::on_posPlus_pressed()
{
   double newPos = m_position + m_step;

   pcf::IndiProperty ip(pcf::IndiProperty::Number);
   
   ip.setDevice(m_stageName);
   if(m_filterWheel)
   {
      ip.setName("filter");
   }
   else
   {
      ip.setName("position");
   }
   ip.add(pcf::IndiElement("target"));
   ip["target"] = newPos;
   
   sendNewProperty(ip);
}

void stage::on_posStepMulTen_pressed()
{
   m_step *= 10.0;
   ui.stepSize->setText(QString::number(m_step));
}

void stage::on_posStepDivTen_pressed()
{
   m_step /= 10.0;
   ui.stepSize->setText(QString::number(m_step));
}

void stage::on_home_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_stageName);
   ipFreq.setName("home");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void stage::on_stop_pressed()
{
   pcf::IndiProperty ipFreq(pcf::IndiProperty::Switch);
   
   ipFreq.setDevice(m_stageName);
   ipFreq.setName("stop");
   ipFreq.add(pcf::IndiElement("request"));
   ipFreq["request"].setSwitchState(pcf::IndiElement::On);
    
   sendNewProperty(ipFreq);   
}

void stage::setPointEditTimerOut()
{
   m_setPointEditing = STOPPED;
   ui.setPoint->setCurrentText(m_setPoint.c_str());
   update();
}

void stage::paintEvent(QPaintEvent * e)
{
   if(m_setPointEditing == STARTED)
   {
      ui.setPoint->setProperty("isStatus", false);
      ui.setPoint->setProperty("isEditing", true);
      style()->unpolish(ui.setPoint);
   }
   else 
   {
      ui.setPoint->setProperty("isEditing", false);
      ui.setPoint->setProperty("isStatus", true);
      style()->unpolish(ui.setPoint);
   }
   
   QWidget::paintEvent(e);
}

} //namespace xqt
   
#include "moc_stage.cpp"

#endif
