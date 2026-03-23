
#ifndef polarimetry_hpp
#define polarimetry_hpp

#include <iostream>
#include "ui_polarimetry.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

class polarimetry : public xWidget
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
   bool m_tracking{false};

   bool m_hwptrackFsmOk{ false };
   bool m_hwpseqFsmOk{ false };


   void setBold(QLabel*, bool);


public:
   explicit polarimetry( QWidget * Parent = 0,
                    Qt::WindowFlags f = Qt::WindowFlags()
                  );

   ~polarimetry();

   void subscribe();

   virtual void onConnect();
   virtual void onDisconnect();

   void handleDefProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);

   void handleSetProperty( const pcf::IndiProperty & ipRecv /**< [in] the property which has changed*/);


public slots:
   void updateGUI();

   void on_buttonStartSequence_pressed();
   void on_buttonLastCycle_clicked(bool);
   void on_buttonStopSequence_pressed();

signals:

   void doUpdateGUI();

private:

   Ui::polarimetry ui;
};

polarimetry::polarimetry(
                QWidget * Parent,
                Qt::WindowFlags f) : xWidget(Parent, f)
{
   ui.setupUi(this);

   setWindowTitle(QString("Polarimetry (disconnected)"));

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
   setXwFont(ui.labelTracking);
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

   ui.hwptrackFsm->highlightChanges(false);
   ui.hwpseqFsm->highlightChanges(false);
   ui.stagePolRotFsm->highlightChanges(false);
   ui.stagePolLinFsm->highlightChanges(false);

   ui.labelTracking->setVisible(false);
   ui.labelTracking->setText(QString("SYNCHRO_ADI"));

   ui.labelHwptrack->setText(QString("hwptrack"));
   ui.hwptrackFsm->device("hwptrack");

   ui.labelHwpseq->setText(QString("hwpsequence"));
   ui.hwpseqFsm->device("hwpsequence");

   ui.labelStagePolRot->setText(QString("stagepolrot"));
   ui.stagePolRotFsm->device("stagepolrot");

   ui.labelStagePolLin->setText(QString("stagepollin"));
   ui.stagePolLinFsm->device("stagepollin");


   ui.buttonLastCycle->setCheckable(true);
   ui.buttonLastCycle->setProperty("isHighlightButton", true);

   ui.entryHwpAngle->setup("hwptrack", "hwp_position", statusEntry::FLOAT, "HWP target", "°");
   ui.entryHwpAngle->format("%.01f");
   ui.entryHwpAngle->setStretch(0, 2, 1);

   ui.sliderTracking->setup("hwptrack", "tracking", "toggle", "HWP Tracking");
   ui.sliderTracking->setStretch(0, 1, 3, true, false);

   ui.comboHwpLin->setup("stagepollin", "", "", "HWP lin. stage", "");
   ui.comboHwpLin->ctrlWidget(nullptr);

   ui.entryNumCycles->setup("hwpsequence", "numCycles", statusEntry::INT, "Num. cycles", "");
   ui.entryNumCycles->setStretch(0, 2, 1);

   ui.entryTimePerPos->setup("hwpsequence", "timePerPos", statusEntry::FLOAT, "Time per pos.", "s");
   ui.entryTimePerPos->format("%.01f");
   ui.entryTimePerPos->setStretch(0, 2, 1);


   connect(this, SIGNAL(doUpdateGUI()), this, SLOT(updateGUI()));

   onDisconnect();
}

polarimetry::~polarimetry()
{
   if(m_parent) m_parent->unsubscribe(this);
}

void polarimetry::subscribe()
{
   if(!m_parent) return;

   m_parent->addSubscriberProperty(this, "hwptrack", "fsm");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_tracking_offset");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position_actual");
   m_parent->addSubscriberProperty(this, "hwptrack", "hwp_position_name");
   m_parent->addSubscriberProperty(this, "hwptrack", "tracking");

   m_parent->addSubscriberProperty(this, "hwpsequence", "fsm");
   m_parent->addSubscriberProperty(this, "hwpsequence", "sequence");
   m_parent->addSubscriberProperty(this, "hwpsequence", "hwpPosIndex");
   m_parent->addSubscriberProperty(this, "hwpsequence", "curCycle");
   m_parent->addSubscriberProperty(this, "hwpsequence", "lastCycle");
   m_parent->addSubscriberProperty(this, "hwpsequence", "numCycles");

   m_parent->addSubscriber(ui.hwptrackFsm);
   m_parent->addSubscriber(ui.hwpseqFsm);
   m_parent->addSubscriber(ui.stagePolRotFsm);
   m_parent->addSubscriber(ui.stagePolLinFsm);
   m_parent->addSubscriber(ui.entryHwpAngle);
   m_parent->addSubscriber(ui.sliderTracking);
   m_parent->addSubscriber(ui.comboHwpLin);
   m_parent->addSubscriber(ui.entryNumCycles);
   m_parent->addSubscriber(ui.entryTimePerPos);

   return;
}

void polarimetry::onConnect()
{

   setWindowTitle(QString("Polarimetry"));
   ui.hwptrackFsm->onConnect();
   ui.hwpseqFsm->onConnect();
   ui.stagePolRotFsm->onConnect();
   ui.stagePolLinFsm->onConnect();


   setBold(ui.hwpSetAngle, true);
   setBold(ui.hwpTrackingOffset, true);
   setBold(ui.hwpActualAngle, true);
   setBold(ui.hwpAngleName, true);

   ui.entryHwpAngle->onConnect();
   ui.sliderTracking->onConnect();
   ui.comboHwpLin->onConnect();
   ui.entryNumCycles->onConnect();
   ui.entryTimePerPos->onConnect();

   ui.entryHwpAngle->setEnabled(true);
   ui.sliderTracking->setEnabled(true);
   ui.comboHwpLin->setEnabled(false);
   ui.entryNumCycles->setEnabled(true);
   ui.negOneLabel->setEnabled(true);
   ui.entryTimePerPos->setEnabled(true);

   ui.buttonStartSequence->setEnabled(true);
   ui.buttonLastCycle->setEnabled(true);
   ui.buttonStopSequence->setEnabled(true);
}


void polarimetry::onDisconnect()
{
   setWindowTitle(QString("Polarimetry (disconnected)"));

   ui.hwptrackFsm->onDisconnect();
   ui.hwpseqFsm->onDisconnect();
   ui.stagePolRotFsm->onDisconnect();
   ui.stagePolLinFsm->onDisconnect();

   ui.hwpSetAngle->setText(QString("---"));
   ui.hwpTrackingOffset->setText(QString("---"));
   ui.hwpActualAngle->setText(QString("---"));
   ui.hwpAngleName->setText(QString("---"));
   setBold(ui.hwpSetAngle, false);
   setBold(ui.hwpTrackingOffset, false);
   setBold(ui.hwpActualAngle, false);
   setBold(ui.hwpAngleName, false);

   ui.hwpPosIndex->setText(QString("---"));
   ui.cycleNumStatus->setText(QString("---"));
   setBold(ui.hwpPosIndex, false);
   setBold(ui.cycleNumStatus, false);

   ui.entryHwpAngle->onDisconnect();
   ui.sliderTracking->onDisconnect();
   ui.comboHwpLin->onDisconnect();
   ui.entryNumCycles->onDisconnect();
   ui.entryTimePerPos->onDisconnect();


   ui.entryHwpAngle->setEnabled(false);
   ui.sliderTracking->setEnabled(false);
   ui.comboHwpLin->setEnabled(false);
   ui.entryNumCycles->setEnabled(false);
   ui.negOneLabel->setEnabled(false);
   ui.entryTimePerPos->setEnabled(false);

   ui.buttonStartSequence->setEnabled(false);
   ui.buttonLastCycle->setEnabled(false);
   ui.buttonStopSequence->setEnabled(false);

   multiIndiSubscriber::onDisconnect();
}

void polarimetry::handleDefProperty( const pcf::IndiProperty & ipRecv)
{
   return handleSetProperty(ipRecv);
}

void polarimetry::handleSetProperty( const pcf::IndiProperty & ipRecv)
{
   if (ipRecv.getDevice() == "hwptrack")
   {
      if (ipRecv.getName() == "fsm")
      {
         if (ipRecv.find("state"))
         {
            std::string fsmString = ipRecv["state"].get<std::string>();
            m_hwptrackFsmOk = fsmString == "READY" || fsmString == "OPERATING";
         }
      }
      else if (ipRecv.getName() == "hwp_position")
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
      else if (ipRecv.getName() == "tracking")
      {
         if (ipRecv.find("toggle"))
         {
            m_tracking = ipRecv["toggle"].getSwitchState() == pcf::IndiElement::On;
         }
      }
   }
   else if (ipRecv.getDevice() == "hwpsequence")
   {
      if (ipRecv.getName() == "fsm")
      {
         if (ipRecv.find("state"))
         {
            std::string fsmString = ipRecv["state"].get<std::string>();
            m_hwpseqFsmOk = fsmString == "READY" || fsmString == "OPERATING";
         }
      }
      else if (ipRecv.getName() == "sequence")
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


void polarimetry::updateGUI()
{
   if (m_hwptrackFsmOk)
   {
      ui.hwpSetAngle->setText(QString("%1°").arg(m_hwpSetAngle, 0, 'f', 1));
      ui.hwpTrackingOffset->setText(QString("%1°").arg(m_hwpTrackingOffset, 0, 'f', 1));
      ui.hwpActualAngle->setText(QString("%1°").arg(m_hwpActualAngle, 0, 'f', 1));
      ui.hwpAngleName->setText(QString(m_hwpAngleName.c_str()));
   }
   else
   {
      ui.hwpSetAngle->setText(QString("---"));
      ui.hwpTrackingOffset->setText(QString("---"));
      ui.hwpActualAngle->setText(QString("---"));
      ui.hwpAngleName->setText(QString("---"));
   }

   setBold(ui.hwpSetAngle, m_hwptrackFsmOk);
   setBold(ui.hwpTrackingOffset, m_hwptrackFsmOk);
   setBold(ui.hwpActualAngle, m_hwptrackFsmOk);
   setBold(ui.hwpAngleName, m_hwptrackFsmOk);


   // disable things that we shouldn't change while sequencing
   ui.entryHwpAngle->setEnabled(!m_sequencing && m_hwptrackFsmOk);
   // we actually don't want to disable the toggleSlider because it will appear "off" even if tracking is on
   if (m_sequencing || !m_hwptrackFsmOk)
   {
      ui.sliderTracking->setAttribute(Qt::WA_TransparentForMouseEvents, true);
      ui.sliderTracking->setFocusPolicy(Qt::NoFocus);
   }
   else
   {
      ui.sliderTracking->setAttribute(Qt::WA_TransparentForMouseEvents, false);
      ui.sliderTracking->setFocusPolicy(Qt::StrongFocus);
   }
   ui.sliderTracking->setLabelEnabled(!m_sequencing && m_hwptrackFsmOk);
   ui.labelTracking->setVisible(m_tracking);
   ui.labelTracking->setEnabled(!m_sequencing && m_hwptrackFsmOk);
   ui.entryNumCycles->setEnabled(!m_sequencing && m_hwpseqFsmOk);
   ui.negOneLabel->setEnabled(!m_sequencing && m_hwpseqFsmOk);
   ui.entryTimePerPos->setEnabled(!m_sequencing && m_hwpseqFsmOk);

   ui.buttonStartSequence->setVisible(!m_sequencing);
   ui.buttonStartSequence->setEnabled(!m_sequencing && m_hwpseqFsmOk && m_hwptrackFsmOk);

   ui.buttonStopSequence->setVisible(m_sequencing);
   ui.buttonStopSequence->setEnabled(m_sequencing && m_hwpseqFsmOk);

   ui.buttonLastCycle->setVisible(m_sequencing);
   ui.buttonLastCycle->setEnabled(m_sequencing && m_hwpseqFsmOk);

   if (m_sequencing)
   {
      ui.hwpPosIndex->setText(QString("%1 / 4").arg(m_hwpPosIndex + 1, 0, 'd'));

      if (m_numCycles > 0)
      {
         ui.cycleNumStatus->setText(QString("%1 / %2").arg(m_curCycle, 0, 'd').arg(m_numCycles, 0, 'd'));
      }
      else
      {
         ui.cycleNumStatus->setText(QString("%1").arg(m_curCycle, 0, 'd'));
      }
   }
   else
   {
      ui.hwpPosIndex->setText(QString("---"));

      ui.cycleNumStatus->setText(QString("---"));
   }

   setBold(ui.hwpPosIndex, m_sequencing);
   setBold(ui.cycleNumStatus, m_sequencing);


} //updateGUI()

void polarimetry::on_buttonStartSequence_pressed()
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

void polarimetry::on_buttonStopSequence_pressed()
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

void polarimetry::on_buttonLastCycle_clicked(bool checked)
{
   pcf::IndiProperty ip(pcf::IndiProperty::Switch);
   ip.setDevice("hwpsequence");
   ip.setName("lastCycle");
   ip.add(pcf::IndiElement("toggle"));
   ip["toggle"] = checked ? pcf::IndiElement::On : pcf::IndiElement::Off;
   sendNewProperty(ip);

   emit doUpdateGUI();
   return;

}

void polarimetry::setBold(QLabel *label, bool onoff)
{
   QFont font = label->font();
   font.setBold(onoff);
   label->setFont(font);
}

} //namespace xqt

#include "moc_polarimetry.cpp"

#endif
