
#ifndef hwpSequencer_hpp
#define hwpSequencer_hpp

#include <iostream>
#include "ui_hwpSequencer.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

class hwpSequencer : public xWidget
{
   Q_OBJECT

protected:

   std::string m_appState;

   float m_hwpSetAngle{0};
   float m_hwpTrackingOffset{0};
   float m_hwpActualAngle{0};
   std::string m_hwpAngleName{""};

   bool m_sequencing{false};
   int m_hwpPosIndex{0};
   int m_curCycle{0};


public:
   explicit hwpSequencer( QWidget * Parent = 0,
                    Qt::WindowFlags f = Qt::WindowFlags()
                  );

   ~hwpSequencer();

   void subscribe();

   virtual void onConnect();
   virtual void onDisconnect();

   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);


public slots:
   void updateGUI();

   // void on_comboSelectPolLin_activated(int);

   void on_buttonStartSequence_pressed();
   void on_buttonLastCycle_clicked(bool);
   void on_buttonStopSequence_pressed();

signals:

   void doUpdateGUI();

private:

   Ui::hwpSequencer ui;
};

hwpSequencer::hwpSequencer(
                QWidget * Parent,
                Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);
   //ui.labelDMName->setText(m_dmName.c_str());

   setWindowTitle(QString("HWP Sequencer (disconnected)"));

   setXwFont(ui.labelAngle);
   setXwFont(ui.labelOffset);
   setXwFont(ui.labelActual);
   setXwFont(ui.labelName);
   setXwFont(ui.labelHwp);

   setXwFont(ui.hwpSetAngle);
   setXwFont(ui.hwpTrackingOffset);
   setXwFont(ui.hwpActualAngle);
   setXwFont(ui.hwpAngleName);

   setXwFont(ui.labelCycleStatus);
   setXwFont(ui.hwpPosIndex);
   setXwFont(ui.labelCycleNum);
   setXwFont(ui.cycleNumStatus);

   setXwFont(ui.buttonStartSequence);
   setXwFont(ui.buttonLastCycle);
   setXwFont(ui.buttonStopSequence);
   setXwFont(ui.negOneLabel);

   ui.buttonLastCycle->setCheckable(true);
   ui.buttonLastCycle->setProperty("isHighlightButton", true);


   ui.entryHwpAngle->setup("hwptrack", "hwp_position", statusEntry::FLOAT, "HWP target", "°");
   ui.entryHwpAngle->format("%.01f");
   ui.entryHwpAngle->readOnly(false);
   ui.entryHwpAngle->setStretch(0, 2, 1);

   ui.sliderTracking->setup("hwptrack", "tracking", "toggle", "Tracking");
   ui.sliderTracking->setStretch(0, 1, 3, true, false);

   // ui.comboHwpLin->setup("stagepollin", "", "", "HWP lin. stage", "");
   // ui.comboHwpLin->ctrlWidget(nullptr);

   ui.entryNumCycles->setup("hwpsequence", "numCycles", statusEntry::INT, "Num. cycles", "");
   ui.entryNumCycles->readOnly(false);
   ui.entryNumCycles->setStretch(0, 2, 1);

   ui.entryTimePerPos->setup("hwpsequence", "timePerPos", statusEntry::FLOAT, "Time per pos.", "s");
   ui.entryTimePerPos->format("%.01f");
   ui.entryTimePerPos->readOnly(false);
   ui.entryTimePerPos->setStretch(0, 2, 1);


   connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

   onDisconnect();
}

hwpSequencer::~hwpSequencer()
{
   if(m_parent) m_parent->unsubscribe(this);
}

void hwpSequencer::subscribe()
{
   if(!m_parent) return;

   m_parent->addSubscriberProperty(this, "hwptrack", "fsm");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_tracking_offset");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position_actual");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position_name");

   m_parent->addSubscriberProperty(this, "hwpsequence", "sequence");
   m_parent->addSubscriberProperty(this, "hwpsequence", "hwpPosIndex");
   m_parent->addSubscriberProperty(this, "hwpsequence", "curCycle");
   m_parent->addSubscriberProperty(this, "hwpsequence", "lastCycle");

   m_parent->addSubscriber(ui.entryHwpAngle);
   m_parent->addSubscriber(ui.sliderTracking);
   // m_parent->addSubscriber(ui.comboHwpLin);
   m_parent->addSubscriber(ui.entryNumCycles);
   m_parent->addSubscriber(ui.entryTimePerPos);

   return;
}

void hwpSequencer::onConnect()
{

   setWindowTitle(QString("HWP Sequencer"));
   ui.entryHwpAngle->onConnect();
   ui.sliderTracking->onConnect();
   // ui.comboHwpLin->onConnect();
   ui.entryNumCycles->onConnect();
   ui.entryTimePerPos->onConnect();

   ui.entryHwpAngle->setEnabled(true);
   ui.sliderTracking->setEnabled(true);
   // ui.comboHwpLin->setEnabled(false);
   ui.entryNumCycles->setEnabled(true);
   ui.entryTimePerPos->setEnabled(true);

   ui.buttonStartSequence->setEnabled(true);
   ui.buttonLastCycle->setEnabled(true);
   ui.buttonStopSequence->setEnabled(true);
}

void hwpSequencer::onDisconnect()
{
   //ui.labelDMName->setEnabled(false);

   setWindowTitle(QString("HWP Sequencer (disconnected)"));

   ui.entryHwpAngle->onDisconnect();
   ui.sliderTracking->onDisconnect();
   // ui.comboHwpLin->onDisconnect();
   ui.entryNumCycles->onDisconnect();
   ui.entryTimePerPos->onDisconnect();


   ui.entryHwpAngle->setEnabled(false);
   ui.sliderTracking->setEnabled(false);
   // ui.comboHwpLin->setEnabled(false);
   ui.entryNumCycles->setEnabled(false);
   ui.entryTimePerPos->setEnabled(false);

   ui.buttonStartSequence->setEnabled(false);
   ui.buttonLastCycle->setEnabled(false);
   ui.buttonStopSequence->setEnabled(false);

   multiIndiSubscriber::onDisconnect();
}

void hwpSequencer::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void hwpSequencer::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   if (ipRecv.getDevice() == "hwptrack")
   {
      if (ipRecv.getName() == "hwp_position")
      {
         if (ipRecv.find("current"))
         {
            m_hwpSetAngle = ipRecv["current"].get<float>();
         }
      }
      else if (ipRecv.getName() == "hwp_tracking_offset")
      {
         if (ipRecv.find("value"))
         {
            m_hwpTrackingOffset = ipRecv["value"].get<float>();
         }
      }
      else if (ipRecv.getName() == "hwp_position_actual")
      {
         if (ipRecv.find("value"))
         {
            m_hwpActualAngle = ipRecv["value"].get<float>();
         }
      }
      else if (ipRecv.getName() == "hwp_position_name")
      {
         if (ipRecv.find("value"))
         {
            m_hwpAngleName = ipRecv["value"].get<std::string>();
         }
      }
   }
   else if (ipRecv.getDevice() == "hwpsequence")
   {
      if (ipRecv.getName() == "sequence")
      {
         if (ipRecv.find("toggle"))
         {
            //m_sequencing = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
         }
      }
      else if (ipRecv.getName() == "curCycle")
      {
         if (ipRecv.find("value"))
         {
            m_curCycle = ipRecv["value"].get<int>();
         }
      }
      else if (ipRecv.getName() == "hwpPosIndex")
      {
         if (ipRecv.find("value"))
         {
            m_hwpPosIndex = ipRecv["value"].get<int>();
         }
      }
      else if (ipRecv.getName() == "lastCycle")
      {
         if (ipRecv.find("toggle"))
         {
            ui.buttonLastCycle->setChecked(ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On);
         }
      }
   }
   else
   {
      return;
   }



   emit doUpdateGUI();
}


void hwpSequencer::updateGUI()
{
   ui.hwpSetAngle->setText(QString("%1°").arg(m_hwpSetAngle, 0, 'f', 1));
   ui.hwpTrackingOffset->setText(QString("%1°").arg(m_hwpTrackingOffset, 0, 'f', 1));
   ui.hwpActualAngle->setText(QString("%1°").arg(m_hwpActualAngle, 0, 'f', 1));
   ui.hwpAngleName->setText(QString(m_hwpAngleName.c_str()));

   // disable things that we shouldn't change while sequencing
   ui.entryHwpAngle->setEnabled(!m_sequencing);
   ui.sliderTracking->setEnabled(!m_sequencing);
   ui.entryNumCycles->setEnabled(!m_sequencing);
   ui.negOneLabel->setEnabled(!m_sequencing);
   ui.entryTimePerPos->setEnabled(!m_sequencing);
   ui.buttonStartSequence->setVisible(!m_sequencing);
   ui.buttonStartSequence->setEnabled(!m_sequencing);
   ui.buttonStopSequence->setVisible(m_sequencing);
   ui.buttonStopSequence->setEnabled(m_sequencing);
   ui.buttonLastCycle->setVisible(m_sequencing);
   ui.buttonLastCycle->setEnabled(m_sequencing);

   if (m_sequencing)
   {
      ui.hwpPosIndex->setText(QString("%1 / 4").arg(m_hwpPosIndex, 0, 'd'));
      QFont font = ui.hwpPosIndex->font();
      font.setBold(true);
      ui.hwpPosIndex->setFont(font);

      ui.cycleNumStatus->setText(QString("%1").arg(m_curCycle, 0, 'd'));
      font = ui.cycleNumStatus->font();
      font.setBold(true);
      ui.cycleNumStatus->setFont(font);
   }
   else
   {
      ui.hwpPosIndex->setText(QString("---"));
      QFont font = ui.hwpPosIndex->font();
      font.setBold(false);
      ui.hwpPosIndex->setFont(font);

      ui.cycleNumStatus->setText(QString("---"));
      font = ui.cycleNumStatus->font();
      font.setBold(false);
      ui.cycleNumStatus->setFont(font);
   }


} //updateGUI()

void hwpSequencer::on_buttonStartSequence_pressed()
{
   m_sequencing = true;
   std::cerr << "Pretend mode: starting sequencing" << std::endl;

   // pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   // ip.setDevice("hwpsequence");
   // ip.setName("sequence");
   // ip.add(pcf::IndiElement("toggle"));
   // ip["toggle"] = pcf::IndiElement::On;
   // sendNewProperty(ip);

   emit doUpdateGUI();
   return;
}

void hwpSequencer::on_buttonStopSequence_pressed()
{
   m_sequencing = false;
   std::cerr << "Pretend mode: stopping sequencing" << std::endl;

   // pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   // ip.setDevice("hwpsequence");
   // ip.setName("sequence");
   // ip.add(pcf::IndiElement("toggle"));
   // ip["toggle"] = pcf::IndiElement::Off;
   // sendNewProperty(ip);

   emit doUpdateGUI();
   return;
}

void hwpSequencer::on_buttonLastCycle_clicked(bool checked)
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   ip.setDevice("hwpsequence");
   ip.setName("lastCycle");
   ip.add(pcf::IndiElement("toggle"));
   ip["toggle"] = checked ? pcf::IndiElement::On : pcf::IndiElement::Off;
   sendNewProperty(ip);

   std::cerr << "lastCycle toggled to: " << (checked ? "On" : "Off") << std::endl;

   emit doUpdateGUI();
   return;
}


} //namespace xqt

#include "moc_hwpSequencer.cpp"

#endif
