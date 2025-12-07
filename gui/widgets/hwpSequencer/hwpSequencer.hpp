
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
   int m_numCycles{-1};


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

   setWindowTitle(QString("HWP Sequencer (disconnected)"));

   setXwFont(ui.labelHwptrack);
   setXwFont(ui.labelHwpseq);
   setXwFont(ui.labelStagePolRot);
   setXwFont(ui.labelStagePolLin);

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


   ui.labelHwptrack->setText(QString("hwptrack"));
   ui.hwptrackFsm->device("hwptrack");
   
   ui.labelHwpseq->setText(QString("hwpsequence"));
   ui.hwpseqFsm->device("hwpsequence");
   
   ui.labelStagePolRot->setText(QString("stagepolrot"));
   ui.stagePolRotFsm->device("stagepolrot");

   ui.labelStagePolLin->setText(QString("stagepollin"));
   ui.labelStagePolLin->setEnabled(false);
   ui.stagePolLinFsm->setEnabled(false);
   

   ui.buttonLastCycle->setCheckable(true);
   ui.buttonLastCycle->setProperty("isHighlightButton", true);

   ui.entryHwpAngle->setup("hwptrack", "hwp_position", statusEntry::FLOAT, "HWP target", "°");
   ui.entryHwpAngle->format("%.01f");
   ui.entryHwpAngle->setStretch(0, 2, 1);

   ui.sliderTracking->setup("hwptrack", "tracking", "toggle", "Tracking");
   ui.sliderTracking->setStretch(0, 1, 3, true, false);

   // ui.comboHwpLin->setup("stagepollin", "", "", "HWP lin. stage", "");
   // ui.comboHwpLin->ctrlWidget(nullptr);

   ui.entryNumCycles->setup("hwpsequence", "numCycles", statusEntry::INT, "Num. cycles", "");
   ui.entryNumCycles->setStretch(0, 2, 1);

   ui.entryTimePerPos->setup("hwpsequence", "timePerPos", statusEntry::FLOAT, "Time per pos.", "s");
   ui.entryTimePerPos->format("%.01f");
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

   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_tracking_offset");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position_actual");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position_name");

   m_parent->addSubscriberProperty(this, "hwpsequence", "sequence");
   m_parent->addSubscriberProperty(this, "hwpsequence", "hwpPosIndex");
   m_parent->addSubscriberProperty(this, "hwpsequence", "curCycle");
   m_parent->addSubscriberProperty(this, "hwpsequence", "lastCycle");
   m_parent->addSubscriberProperty(this, "hwpsequence", "numCycles");

   m_parent->addSubscriber(ui.hwptrackFsm);
   m_parent->addSubscriber(ui.hwpseqFsm);
   m_parent->addSubscriber(ui.stagePolRotFsm);
   // m_parent->addSubscriber(ui.stagePolLinFsm);
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
   ui.hwptrackFsm->onConnect();
   ui.hwpseqFsm->onConnect();
   ui.stagePolRotFsm->onConnect();
   // ui.stagePolLinFsm->onConnect();

   ui.entryHwpAngle->onConnect();
   ui.sliderTracking->onConnect();
   // ui.comboHwpLin->onConnect();
   ui.entryNumCycles->onConnect();
   ui.entryTimePerPos->onConnect();

   ui.entryHwpAngle->setEnabled(true);
   ui.sliderTracking->setEnabled(true);
   // ui.comboHwpLin->setEnabled(false);
   ui.entryNumCycles->setEnabled(true);
   ui.negOneLabel->setEnabled(true);
   ui.entryTimePerPos->setEnabled(true);

   ui.buttonStartSequence->setEnabled(true);
   ui.buttonLastCycle->setEnabled(true);
   ui.buttonStopSequence->setEnabled(true);
}

void hwpSequencer::onDisconnect()
{
   //ui.labelDMName->setEnabled(false);

   setWindowTitle(QString("HWP Sequencer (disconnected)"));

   ui.hwptrackFsm->onDisconnect();
   ui.hwpseqFsm->onDisconnect();
   ui.stagePolRotFsm->onDisconnect();
   // ui.stagePolLinFsm->onDisconnect();

   ui.entryHwpAngle->onDisconnect();
   ui.sliderTracking->onDisconnect();
   // ui.comboHwpLin->onDisconnect();
   ui.entryNumCycles->onDisconnect();
   ui.entryTimePerPos->onDisconnect();


   ui.entryHwpAngle->setEnabled(false);
   ui.sliderTracking->setEnabled(false);
   // ui.comboHwpLin->setEnabled(false);
   ui.entryNumCycles->setEnabled(false);
   ui.negOneLabel->setEnabled(false);
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
            m_sequencing = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
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
      else if (ipRecv.getName() == "numCycles")
      {
         if (ipRecv.find("current"))
         {
            m_numCycles = ipRecv["current"].get<int>();
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
      ui.hwpPosIndex->setText(QString("%1 / 4").arg(m_hwpPosIndex + 1, 0, 'd'));
      QFont font = ui.hwpPosIndex->font();
      font.setBold(true);
      ui.hwpPosIndex->setFont(font);

      if (m_numCycles > 0)
      {
         ui.cycleNumStatus->setText(QString("%1 / %2").arg(m_curCycle, 0, 'd').arg(m_numCycles, 0, 'd'));
      }
      else
      {
         ui.cycleNumStatus->setText(QString("%1").arg(m_curCycle, 0, 'd'));
      }
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
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   ip.setDevice("hwpsequence");
   ip.setName("sequence");
   ip.add(pcf::IndiElement("toggle"));
   ip["toggle"] = pcf::IndiElement::On;
   sendNewProperty(ip);

   emit doUpdateGUI();
   return;
}

void hwpSequencer::on_buttonStopSequence_pressed()
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   ip.setDevice("hwpsequence");
   ip.setName("sequence");
   ip.add(pcf::IndiElement("toggle"));
   ip["toggle"] = pcf::IndiElement::Off;
   sendNewProperty(ip);

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
