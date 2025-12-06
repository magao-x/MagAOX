
#ifndef hwpSequencer_hpp
#define hwpSequencer_hpp

#include "ui_hwpSequencer.h"

#include "../xWidgets/xWidget.hpp"

namespace xqt
{

class hwpSequencer : public xWidget
{
   Q_OBJECT

protected:

   std::string m_appState;

   float m_hwpSetAngle;
   float m_hwpTrackingOffset;
   float m_hwpActualAngle;
   std::string m_hwpAngleName;


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

   // void on_buttonStartSequence_pressed();
   // void on_buttonLastCycle_pressed();
   // void on_buttonStopSequence_pressed();

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

   setXwFont(ui.buttonStartSequence);
   setXwFont(ui.buttonLastCycle);
   setXwFont(ui.buttonStopSequence);
   // setXwFont(ui.comboSelectPolLin);



   ui.sliderTracking->setup("hwptrack", "tracking", "toggle", "");
   ui.sliderTracking->setStretch(0, 0, 3, true, true);

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

   m_parent->addSubscriber(ui.sliderTracking);

   return;
}

void hwpSequencer::onConnect()
{

   setWindowTitle(QString("HWP Sequencer"));
   ui.sliderTracking->onConnect();
}

void hwpSequencer::onDisconnect()
{
   //ui.labelDMName->setEnabled(false);

   setWindowTitle(QString("HWP Sequencer (disconnected)"));
   
   ui.sliderTracking->onDisconnect();

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

   }
   else if (ipRecv.getDevice() == "stagepollin")
   {

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


} //updateGUI()

// void hwpSequencer::on_buttonStartSequence_pressed()
// {

// }

// void hwpSequencer::on_buttonStopSequence_pressed()
// {

// }


} //namespace xqt

#include "moc_hwpSequencer.cpp"

#endif
